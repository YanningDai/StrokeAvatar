using System;
using Unity.MLAgentsExamples;
using UnityEngine;
using Newtonsoft.Json;
using System.IO;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;
using System.Linq;
using Random = UnityEngine.Random;

public class GoalwithImitation : Agent
{
    //private float framerateDataset;
    private float framerate;
    public Transform Target;

    /// <summary>
    /// Objects that need to change color based on ground contact
    /// </summary>
    private GameObject[] meshChangeObject;
    public Material groundedMaterial;
    public Material unGroundedMaterial;

    /// <summary>
    /// Path of the current JSON file
    /// </summary>
    private string jsonDir;

    [Header("User Input Goal")]
    /// <summary>
    /// Randomize target step length (true for training, false for inference)
    /// </summary>
    public bool randomizeWalkLengthEachEpisode;
    /// <summary>
    /// Randomize target speed (true for training, false for inference)
    /// </summary>
    public bool randomizeWalkSpeedEachEpisode;
    /// <summary>
    /// User-input target step length for two steps (dataset dependent, ~0.7-2)
    /// </summary>
    public float lengthInput;
    /// <summary>
    /// User-input target speed (expected 1-2)
    /// </summary>
    public float velocityInput;

    public int velocityRange; // number of samples for average speed (default 100)

    [Header("Training Data !! Don't Edit !!")]
    /// <summary>
    /// Phase parameter in [0,1]
    /// </summary>
    public float fai;
    /// <summary>
    /// Counter for fai; resets each loop
    /// </summary>
    int i;
    public float averVelocity;
    public Vector3 averComVelocity;
    public float currentLength;

    /// <summary>
    /// Actual target speed
    /// </summary>
    public float velocityGoal;
    public float lengthGoal;
    public float freGoal;
    /// <summary>
    /// Dataset index for reward computation; float for interpolation
    /// </summary>
    public float iDataset;
    public int clipLengthDataset;
    public int clipLengthImitation;
    public string dirPatient;

    [Header("Body Parts")]
    public Transform hips;
    public Transform spine;
    public Transform thighL;
    public Transform shinL;
    public Transform thighR;
    public Transform shinR;
    public Transform footLMesh;
    public Transform footRMesh;
    private JointDriveController m_JdController;
    /// <summary>
    /// Joints used in the simulation (upper limbs excluded)
    /// </summary>
    List<Transform> bodylistInUse;

    private ImuData2 imuData2;
    [HideInInspector] static public bool showtext;
    public StatsRecorder statsRecorder;// tensorboard output

    private List<float> positionlist;
    private List<Vector3> comPositionlist;

    [Header("Reference Body Parts")]
    public Transform hipsRef;
    public Transform spineRef;
    public Transform thighLRef;
    public Transform shinLRef;
    public Transform thighRRef;
    public Transform shinRRef;

    private Vector3 xPositionBeforeI;


    protected override void Awake()
    {
        base.Awake();
        statsRecorder = Academy.Instance.StatsRecorder;// tensorboard output

        meshChangeObject = GameObject.FindGameObjectsWithTag("mesh");

        bodylistInUse = new List<Transform> { hips, thighL, shinL, thighR, shinR, spine };

        // Change this before imitating a different patient
        dirPatient = Application.streamingAssetsPath+"/OutputPatientDataset/walk475.json";

        randomizeWalkSpeedEachEpisode = false;
        imuData2 = JsonConvert.DeserializeObject<ImuData2>(File.ReadAllText(@dirPatient));
        lengthInput = imuData2.stepLength;
        velocityInput = imuData2.aveVelocity;
    }

    public override void Initialize()
    {
        Time.fixedDeltaTime = 1.0f / 30f;// physics step duration

        // Setup each body part
        m_JdController = GetComponent<JointDriveController>();
        m_JdController.SetupBodyPart(hips);
        m_JdController.SetupBodyPart(spine);
        m_JdController.SetupBodyPart(thighL);
        m_JdController.SetupBodyPart(shinL);
        m_JdController.SetupBodyPart(thighR);
        m_JdController.SetupBodyPart(shinR);

        // Reference joints (can be toggled if needed)
        m_JdController.SetupBodyPart(hipsRef);
        m_JdController.SetupBodyPart(spineRef);
        m_JdController.SetupBodyPart(thighLRef);
        m_JdController.SetupBodyPart(shinLRef);
        m_JdController.SetupBodyPart(thighRRef);
        m_JdController.SetupBodyPart(shinRRef);

        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        showtext = true;

        // If training for speed or step length, randomize when specified; otherwise keep fixed
    }



    public override void OnEpisodeBegin()
    {
        ApplyTerrainCurriculum();
        
        i = 0;
        showtext = true;
        // Initialize agent pose
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values) { bodyPart.Reset(bodyPart); };

        // Select reference clip to imitate
        lengthGoal =
            randomizeWalkLengthEachEpisode ? Random.Range(0.42f, 1.2f) : lengthInput;// example fixed: 1.5096f
        velocityGoal =
            randomizeWalkSpeedEachEpisode ? Random.Range(0.4f, 1.6f) : velocityInput;// example fixed: 1.3828f
        freGoal = velocityGoal / lengthGoal;

        jsonDir = dirPatient;// load from patient file directly
        imuData2 = JsonConvert.DeserializeObject<ImuData2>(File.ReadAllText(@jsonDir));// reference for this episode
        // If imitating a specific clip, override here

        // Randomly choose a frame as initial pose; compute frame counts and start indices
        clipLengthDataset = imuData2.walk.Length / 27;
        clipLengthImitation = Mathf.RoundToInt(30 / freGoal);
        Debug.Log("clipLengthDataset: " + clipLengthDataset + " clipLengthImitation: " + clipLengthImitation+ " jsonDir: " + jsonDir);


        i = Random.Range(0, clipLengthImitation-1);// position within imitation clip

        fai = (float)i / clipLengthImitation;// phase used for training
        iDataset = fai * (clipLengthDataset - 1) > (clipLengthDataset - 1) ? (clipLengthDataset - 1) : fai * (clipLengthDataset - 1);

        // Initialize velocity tracking
        positionlist = new List<float>();
        comPositionlist = new List<Vector3>();
        averVelocity = 0;
        xPositionBeforeI = -new Vector3(keyFrameLinearLerp(imuData2.walk, 0, iDataset).x, 0, 0);

        //Debug.Log("i: " + i+ " fai: " + fai+ " iDataset: " + iDataset);
        // Set initial pose
        SetIniPosture(iDataset);

        // Set reference pose
        MotionDataControlRef(imuData2, iDataset);

        RecordPostureFoot();
    }

    void ApplyTerrainCurriculum()
    {
        float terrainValue = Academy.Instance.EnvironmentParameters.GetWithDefault( "my_environment_parameter", 0f );
        StairsController stairs = FindObjectOfType<StairsController>();
        if (stairs != null && stairs.gameObject.activeInHierarchy)
        {
            stairs.ApplyCurriculum(terrainValue);
            return;
        }

        SlopeController slope = FindObjectOfType<SlopeController>();
        if (slope != null && slope.gameObject.activeInHierarchy)
        {
            slope.ApplyCurriculum(terrainValue);
            Debug.Log("Applied slope curriculum with height (cm): " + terrainValue);
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (positionlist.Count > 1) showtext = false;// show text only at start of episode
        GroundTouchVisualize();// ground-contact color change

        // Phase parameter, 1
        sensor.AddObservation(fai);
        //Debug.Log("fai: "+ fai + "lengthGoal"+ (lengthGoal - 1.1f) / 0.6f + " FootHeight(hips): " + (FootHeight(hips) - 0.2f) / 0.8f);
        // Target step length, 1 (range ~0.5-1.7)
        sensor.AddObservation((lengthGoal - 0.5f) / 1.2f);
        // Step length ratio, 1
        if (comPositionlist.Count > clipLengthImitation)
            currentLength = comPositionlist[comPositionlist.Count - 1].x - comPositionlist[comPositionlist.Count - 1 - clipLengthImitation].x;
        else if (comPositionlist.Count == 0)
            currentLength = 0;
        else
            currentLength = comPositionlist[comPositionlist.Count - 1].x - comPositionlist[0].x;
        sensor.AddObservation(currentLength / lengthGoal);
        //Debug.Log("current frame: " + comPositionlist.Count + " currentLength: " + currentLength + " lengthGoal: " + lengthGoal + " Observation:" + (currentLength/ lengthGoal));

        // Hip height above ground, float, 1
        sensor.AddObservation((FootHeight(hips) - 0.2f) / 0.8f);
        //Debug.Log("FootHeight(hips)"+ FootHeight(hipsRef));
        // Speed error, float, 1
        int init;
        init = positionlist.Count > velocityRange ? positionlist.Count - velocityRange : 0;  // starting index for velocity window
        averVelocity = positionlist.Count > 1 ? Mathf.Abs(positionlist[positionlist.Count - 1] - positionlist[init]) / (1.0f / framerate * (positionlist.Count - init - 1)) : m_JdController.bodyPartsDict[hips].rb.velocity.x;
        averComVelocity = positionlist.Count > 1 ? (comPositionlist[positionlist.Count - 1] - comPositionlist[init]) / (1.0f / framerate * (positionlist.Count - init - 1)) : m_JdController.bodyPartsDict[hips].rb.velocity;
        averComVelocity = new Vector3(averComVelocity.x, 0, averComVelocity.z);

        sensor.AddObservation(averComVelocity.z / velocityGoal + 0.5f);// centered at 0.5; lateral sway
        //sensor.AddObservation(Vector3.one);
        sensor.AddObservation((velocityGoal - Vector3.Dot(averComVelocity, Vector3.right)) / velocityGoal / 2 / 1.5f + 0.5f);// centered at 0.5

        // Target speed, float, 1 (range ~0.4-1.4)
        sensor.AddObservation((velocityGoal - 0.4f) / 1.0f);
        //Debug.Log("velocityGoal: " + (velocityGoal-0.3f)/1.7f + " speed error 1: " + ((velocityGoal - Vector3.Dot(averComVelocity, Vector3.right)) / velocityGoal / 2/1.5f + 0.5f) + " lateral speed error: " + (averComVelocity.z / velocityGoal + 0.5f));

        // Foot-ground contact flags, bool x2
        sensor.AddObservation(m_JdController.bodyPartsDict[shinL].groundContact.touchingGround);
        sensor.AddObservation(m_JdController.bodyPartsDict[shinR].groundContact.touchingGround);

        // Joint observations

        // hips first: 4 + 3 + 3 = 10
        // Global rotation quaternion, 4
        sensor.AddObservation(hips.rotation);
        // Linear velocity, Vector3, 3
        sensor.AddObservation(m_JdController.bodyPartsDict[hips].rb.velocity / 8f + 0.5f * Vector3.one);
        // Angular velocity, Vector3, 3
        sensor.AddObservation(m_JdController.bodyPartsDict[hips].rb.angularVelocity / 30f + 0.5f * Vector3.one);
        // Root position not added here
        //Debug.Log("m_JdController.bodyPartsDict[hips].rb.velocity: " + (m_JdController.bodyPartsDict[hips].rb.velocity / 8f + 0.5f * Vector3.one) + " m_JdController.bodyPartsDict[hips].rb.angularVelocity: " + (m_JdController.bodyPartsDict[hips].rb.angularVelocity / 30f + 0.5f * Vector3.one));


        //List<Transform> bodylistInUse = new List<Transform> { hips, thighL, shinL, thighR, shinR, spine };
        foreach (var trans in bodylistInUse)
        {
            Unity.MLAgentsExamples.BodyPart bp = m_JdController.bodyPartsDict[trans];

            // Seven joints; each has 4+3+3+1=11 => 77 total; plus three end-effector positions (3*3=9)
            if (trans != hips)
            {
                // Local rotation, quaternion, 4
                sensor.AddObservation(trans.localRotation);
                // Linear velocity, Vector3, 3
                sensor.AddObservation(bp.rb.velocity / 8f + 0.5f * Vector3.one);
                // Angular velocity, Vector3, 3
                sensor.AddObservation(bp.rb.angularVelocity / 30f + 0.5f * Vector3.one);
                // Torque limit, float, 1
                sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
                //Debug.Log("bp.currentStrength / m_JdController.maxJointForceLimit: " + bp.currentStrength / m_JdController.maxJointForceLimit );
                //Debug.Log("m_JdController.bodyPartsDict[transs].rb.velocity: " + (bp.rb.velocity / 8f + 0.5f * Vector3.one) + " m_JdController.bodyPartsDict[trans].rb.angularVelocity: " + (bp.rb.angularVelocity / 30f + 0.5f * Vector3.one));

            }
            if (trans == shinL || trans == shinR || trans == spine)// end-effector position relative to root
                sensor.AddObservation((trans.position - hips.position) / 1.5f + 0.5f * Vector3.one);
            //Debug.Log("trans.position - hips.position: " + ((trans.position - hips.position) / 1.5f + 0.5f * Vector3.one));
        }
    }

    /// <summary>
    /// Action function (shared pattern across scripts)
    /// </summary>
    /// <param name="actions"></param>
    public override void OnActionReceived(ActionBuffers actions)
    {
        var bpDict = m_JdController.bodyPartsDict;
        var n = -1;
        var continuousActions = actions.ContinuousActions;
        //Debug.Log("actions spine: " + new Vector3(continuousActions[0], continuousActions[1], continuousActions[2])+ " actions thighL: " + new Vector3(continuousActions[3], continuousActions[4], 0) + " actions thighR: " + new Vector3(continuousActions[5], continuousActions[6], 0));
        //Debug.Log("actions shinL: " + new Vector3(continuousActions[7], 0, 0) + " actions shinR: " + new Vector3(continuousActions[8], 0, 0));

        bpDict[spine].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], continuousActions[++n]);
        bpDict[thighL].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], 0);
        bpDict[thighR].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], 0);
        bpDict[shinL].SetJointTargetRotation(continuousActions[++n], 0, 0);
        bpDict[shinR].SetJointTargetRotation(continuousActions[++n], 0, 0);

        //update joint strength settings
        bpDict[spine].SetJointStrength(continuousActions[++n]);
        bpDict[thighL].SetJointStrength(continuousActions[++n]);
        bpDict[shinL].SetJointStrength(continuousActions[++n]);
        bpDict[thighR].SetJointStrength(continuousActions[++n]);
        bpDict[shinR].SetJointStrength(continuousActions[++n]);

        //Debug.Log("actions STRENGTH spine thighL shinL: " + new Vector3(continuousActions[9], continuousActions[10], continuousActions[11]) + " actions thighR shinR: " + new Vector3(continuousActions[12], continuousActions[13], 0) );

        positionlist.Add(hips.position.x);
        comPositionlist.Add(comCalculate());

        if (Math.Abs(hips.position.x - Target.position.x) < 1)// stop when close to goal
        {
            SetReward(1);
            Debug.Log("Reached target");
            EndEpisode();
        }

        TotalRewardCalculate(imuData2, iDataset);

        RecordPostureFoot();

        i++;// update after reward calculation
        if (i == clipLengthImitation) i = 0; // wrap back to fai=0
        fai = (float)i / clipLengthImitation;
        iDataset = fai * (clipLengthDataset - 1) > (clipLengthDataset - 1) ? (clipLengthDataset - 1) : fai * (clipLengthDataset - 1);

        MotionDataControlRef(imuData2, iDataset);
    }


    public override void Heuristic(in ActionBuffers actionsOut)
    {

    }
    private void MotionDataControlRef(ImuData2 imuData2, float iDataset)// total 35 dimensions
    {
        // Use this to store intermediate starting position
        if (positionlist.Count == 0)
            m_JdController.bodyPartsDict[hipsRef].PrePosition = xPositionBeforeI;
        else if (i == 0)
            m_JdController.bodyPartsDict[hipsRef].PrePosition = new Vector3(hipsRef.position.x - m_JdController.bodyPartsDict[hipsRef].startingPos.x, 0, 0);

        hipsRef.position = m_JdController.bodyPartsDict[hipsRef].startingPos+ m_JdController.bodyPartsDict[hipsRef].PrePosition + keyFrameLinearLerp(imuData2.walk, 0, iDataset);

        //Debug.Log("positionlist.Count: " + positionlist.Count+ " clipLengthDataset:" + clipLengthDataset+ " Mathf.Floor((float)positionlist.Count / clipLengthDataset):" + Mathf.Floor((float)positionlist.Count / clipLengthDataset));
        //Debug.Log("imuData2.stepLength: " + imuData2.stepLength);

        List<Transform> bodylistInUseRef = new List<Transform> { hipsRef, thighLRef, shinLRef, thighRRef, shinRRef, spineRef };
        int count = 0;
        foreach (var trans in bodylistInUseRef)//q order: x y z w
        {
            Quaternion bodyAngle = keyFrameQLerp(imuData2.walk, 3 + count * 4, iDataset);
            if (trans == hipsRef) trans.rotation = bodyAngle;
            else trans.localRotation = bodyAngle;
            // Do not set rigidbody velocities here; physics will handle them
            count++;
        }

        foreach (var trans in bodylistInUseRef)
        {
            trans.GetComponent<Rigidbody>().velocity = Vector3.zero;
            trans.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        }
    }

    public void TotalRewardCalculate(ImuData2 imuData2, float iDataset)
    {
        // 1: Task achievement (terrain following, direction, speed)
        // 1.1 Hip height matches terrain height
        var delatHeight = (Mathf.Abs(FootHeight(hips) - FootHeight(hipsRef)));// hip height above ground ideally 0.7-0.9
        var delatLateral = (hips.position.z-m_JdController.bodyPartsDict[hips].startingPos.z)-(hipsRef.position.z - m_JdController.bodyPartsDict[hipsRef].startingPos.z);
        float hipHeightReward = 0.5f*(float)Math.Exp(-40 * delatHeight * delatHeight) + 0.5f * (float)Math.Exp(-40 * delatLateral * delatLateral);

        // 1.2 Overall walking direction should face +X; compare hip end-to-start displacement
        Vector3 direction = new Vector3(hips.position.x - m_JdController.bodyPartsDict[hips].startingPos.x, 0, hips.position.z - m_JdController.bodyPartsDict[hips].startingPos.z).normalized;
        float deltadirection = 1 - Mathf.Clamp( Vector3.Dot(direction.normalized, Vector3.right),0,1);
        float forwardReward = (float)Math.Exp(-deltadirection * deltadirection);

        // 1.3 Speed match
        float deltaV = velocityGoal - Vector3.Dot(averComVelocity, Vector3.right);
        float velocityReward = Mathf.Exp(-2.5f * deltaV * deltaV);

        // 1.4 Step length match
        float deltaLength = 1 - Mathf.Clamp(currentLength / lengthGoal, 0, 1);
        float LengthReward = (float)Math.Exp(-deltaLength * deltaLength);

        float goalReward = velocityReward*0.5f + forwardReward*0.2f+ hipHeightReward *0.2f+ LengthReward*0.1f;

        // 2: Walking quality (foot contact, no dragging, distance)
        // 2.1 Foot contact
        float bodyStableReward = 0;
        if (m_JdController.bodyPartsDict[shinL].groundContact.touchingGround && !m_JdController.bodyPartsDict[shinR].groundContact.touchingGround)
        { bodyStableReward = -0.005f; }
        // 2.2 No dragging
        float footDragPenal = FootDrag();// zero when airborne; negative when dragging
        // 2.3 Distance traveled
        float walklengthReward = 0.005f;
        float walkAbilityReward = footDragPenal;


        // 3: Imitation similarity
        // Target frame index: iDataset

        // 3.1 COM position and end-effector positions relative to hip
        // COM
        float delta1 = Vector3.Magnitude(comCalculate() - hips.position - keyFrameLinearLerp(imuData2.comPosition, 0, iDataset));// expected ~0.07-0.1
        float delta2 = Vector3.Magnitude(shinL.position - hips.position - keyFrameLinearLerp(imuData2.endPosition, 0, iDataset));// expected ~0.07
        float delta3 = Vector3.Magnitude(shinR.position - hips.position - keyFrameLinearLerp(imuData2.endPosition, 3, iDataset));
        float delta4 = Vector3.Magnitude(spine.position - hips.position - keyFrameLinearLerp(imuData2.endPosition, 6, iDataset));// expected ~0.01
        float keypositionReward = (float)Math.Exp(-5 * delta2 * delta2 -5* delta3 * delta3 - 30 * delta4 * delta4);
        float compositionReward = (float)Math.Exp(-50 * delta1 * delta1);

        // 3.2 Joint rotations, angular velocities, and linear velocities
        //bodylistInUse = new List<Transform> { hips, thighL, shinL, thighR, shinR, spine };
        //List<Transform> bodylistInUse4 = new List<Transform> { hipsRef, thighLRef, shinLRef, thighRRef, shinRRef, spineRef };
        float keyAngleReward = 0;
        float keyAngleVelociyReward = 0;
        float keyVelocityReward = 0;
        int count = 0;
        foreach (var trans in bodylistInUse)
        {
            Unity.MLAgentsExamples.BodyPart bp = m_JdController.bodyPartsDict[trans];

            // Joint linear velocity in world space (3D)
            float deltaKeyV = Vector3.Magnitude(bp.rb.velocity - keyFrameLinearLerp(imuData2.bodyPartVelocity, count*3, iDataset));
            if (trans != shinL && trans != shinL)
                keyVelocityReward += deltaKeyV * deltaKeyV;
            else
                keyVelocityReward += 0.2f * deltaKeyV * deltaKeyV;

            // Joint angular velocity in world space (3D)
            float deltaKeyAngleV = Vector3.Magnitude(bp.rb.angularVelocity - keyFrameLinearLerp(imuData2.angularVelocity, count * 3, iDataset));
            if (trans != shinL && trans != shinL)
                keyAngleVelociyReward += deltaKeyAngleV * deltaKeyAngleV;
            else
                keyAngleVelociyReward += 0.2f * deltaKeyAngleV * deltaKeyAngleV;

            // Joint rotation (4D quaternion)
            Quaternion bodyAngle = trans == hips ? trans.rotation : trans.localRotation;
            Quaternion datasetAngle = keyFrameQLerp(imuData2.walk, 3 + count * 4, iDataset);
            float deltaKeyAngle = Quaternion.Angle(bodyAngle, datasetAngle) *Mathf.Deg2Rad;
            if (trans != shinL && trans != shinL)
                keyAngleReward += deltaKeyAngle * deltaKeyAngle;
            else
                keyAngleReward += 0.2f*deltaKeyAngle * deltaKeyAngle;

            count++;
        }

        keyVelocityReward = (float)Math.Exp(- 0.1f * keyVelocityReward);
        keyAngleVelociyReward = (float)Math.Exp(-0.1f * keyAngleVelociyReward);
        //Debug.Log("keyAngleReward: " + (float)Math.Exp(-2 * keyAngleReward)+ " BodyAngleRewardinLocalAxis(): "+ BodyAngleRewardinLocalAxis() + " BodyAngleRewardinWorldAxis(): " + BodyAngleRewardinWorldAxis());
        keyAngleReward = BodyAngleRewardinLocalAxis(); // compare in world space if needed: BodyAngleRewardinWorldAxis();

        float keyFrameReward = 0.55f * keyAngleReward + 0.1f * keyAngleVelociyReward + 0.1f * keyVelocityReward + 0.15f * keypositionReward + 0.1f * compositionReward;

        float totalReward = 0.5f * goalReward + 0.5f * keyFrameReward + walkAbilityReward;
        totalReward = (0.5f * goalReward + 0.5f * keyFrameReward + walkAbilityReward) * 2 - 1;
        //totalReward = keyAngleReward*0.2f + walklengthReward + hipHeightReward*0.5f + LengthReward*0.3f;
        //Debug.Log("totalReward: "+ totalReward);
        //Debug.Log("keyAngleReward: " + keyAngleReward + " keyAngleVelociyReward: " + keyAngleVelociyReward + "  keyVelocityReward: " + keyVelocityReward + " totalReward: " + totalReward);

        AddReward(totalReward);

        statsRecorder.Add("1- goal Reward", goalReward, StatAggregationMethod.Average);
        statsRecorder.Add("1.1 hip Height Reward", hipHeightReward, StatAggregationMethod.Average);
        statsRecorder.Add("1.2 velocity Reward", velocityReward, StatAggregationMethod.Average);
        statsRecorder.Add("1.3 forward Reward", forwardReward, StatAggregationMethod.Average);
        statsRecorder.Add("1.4 Length Reward", LengthReward, StatAggregationMethod.Average);

        statsRecorder.Add("2- walk Ability Reward", walkAbilityReward, StatAggregationMethod.Average);
        statsRecorder.Add("2.1 float penalty", bodyStableReward, StatAggregationMethod.Average);
        statsRecorder.Add("2.2 foot Drag Penalty", footDragPenal, StatAggregationMethod.Average);
        statsRecorder.Add("2.3 walk length", walklengthReward, StatAggregationMethod.Average);

        statsRecorder.Add("3- key Frame Reward", keyFrameReward, StatAggregationMethod.Average);
        statsRecorder.Add("3.1 key position Reward", keypositionReward, StatAggregationMethod.Average);
        statsRecorder.Add("3.12 com position Reward", compositionReward, StatAggregationMethod.Average);
        statsRecorder.Add("3.2 key Angle Reward", keyAngleReward, StatAggregationMethod.Average);
        statsRecorder.Add("3.3 key Angle Velocity Reward", keyAngleVelociyReward, StatAggregationMethod.Average);
        statsRecorder.Add("3.4 key Velocity Reward", keyVelocityReward, StatAggregationMethod.Average);
    
        AddReward(0.5f * ComputeStairLegRaiseReward());
    }

    float ComputeStairLegRaiseReward()
    {
        GameObject stepObj = GameObject.Find("Step1");
        if (stepObj == null || !stepObj.activeInHierarchy)
            return 0f;

        Collider stepCol = stepObj.GetComponent<Collider>();
        if (stepCol == null)
            return 0f;

        Transform[] feet = { shinL, shinR };

        bool stepDetected = false;
        float minDx = float.MaxValue;

        float stepX = stepCol.bounds.min.x;  

        foreach (var foot in feet)
        {
            float footX = GetFootTipX(foot);
            float dx = stepX - footX;
            // UnityEngine.Debug.Log($"Foot {foot.name} dx to step: {dx}");
            if (dx >= 0 && dx < minDx) minDx = dx;
        }

        if (minDx <= 0.05f) stepDetected = true;

        if (!stepDetected) return 1f;

        float footClearance = Mathf.Max(
            FootHeight(footLMesh),
            FootHeight(footRMesh)
        );

        float stepHeight = stepCol.transform.localPosition.y;
        float desiredClearance = stepHeight + 0.02f;
        float ratio = Mathf.Clamp01(footClearance / desiredClearance);

        float stairReward = ratio >= 1f ? 1f : ratio * ratio;

        // UnityEngine.Debug.Log("Stair leg raise reward: " + stairReward);
        // UnityEngine.Debug.Log("Step height: " + stepHeight);
        // UnityEngine.Debug.Log("Foot pos: " + footClearance);

        return stairReward;
    }

    float GetFootTipX(Transform foot)
    {
        CapsuleCollider[] caps = foot.GetComponentsInChildren<CapsuleCollider>();
        if (caps == null || caps.Length == 0)
            return foot.position.x;

        foreach (var cap in caps)
        {
            if (cap.direction == 0) // x-axis capsule
            {
                Vector3 localTip = cap.center + Vector3.right * cap.radius;
                return cap.transform.TransformPoint(localTip).x;
            }
        }

        return foot.position.x;
    }

    /// <summary>
    /// Optimize using joint angles in world space
    /// </summary>
    /// <returns></returns>
    public float BodyAngleRewardinWorldAxis()
    {
        float keyAngleReward = 0;
        float deltaKeyAngle;

        deltaKeyAngle = Quaternion.Angle(hips.rotation, hipsRef.rotation) * Mathf.Deg2Rad;
        keyAngleReward += deltaKeyAngle * deltaKeyAngle;
        //Debug.Log("Quaternion.Angle hips: " + Quaternion.Angle(hips.rotation, hipsRef.rotation)+ " deltaKeyAngle: "+ deltaKeyAngle);
        deltaKeyAngle = Quaternion.Angle(shinL.rotation, shinLRef.rotation) * Mathf.Deg2Rad;
        keyAngleReward += deltaKeyAngle * deltaKeyAngle;

        deltaKeyAngle = Quaternion.Angle(shinR.rotation, shinRRef.rotation) * Mathf.Deg2Rad;
        keyAngleReward += deltaKeyAngle * deltaKeyAngle;

        deltaKeyAngle = Quaternion.Angle(thighL.rotation, thighLRef.rotation) * Mathf.Deg2Rad;
        keyAngleReward += deltaKeyAngle * deltaKeyAngle;

        deltaKeyAngle = Quaternion.Angle(thighR.rotation, thighRRef.rotation) * Mathf.Deg2Rad;
        keyAngleReward += deltaKeyAngle * deltaKeyAngle;

        deltaKeyAngle = Quaternion.Angle(spine.rotation, spineRef.rotation) * Mathf.Deg2Rad;
        keyAngleReward += deltaKeyAngle * deltaKeyAngle;

        float outreward = (float)Math.Exp(-2 * keyAngleReward);

        return outreward;
    }

    /// <summary>
    /// Optimize using joint angles in local space
    /// </summary>
    /// <returns></returns>
    public float BodyAngleRewardinLocalAxis()
    {
        float keyAngleReward = 0;
        float deltaKeyAngle;

        deltaKeyAngle = Quaternion.Angle(hips.rotation, hipsRef.rotation) * Mathf.Deg2Rad;
        keyAngleReward += deltaKeyAngle * deltaKeyAngle;
        //Debug.Log("Quaternion.Angle hips: " + Quaternion.Angle(hips.rotation, hipsRef.rotation)+ " deltaKeyAngle: "+ deltaKeyAngle);
        deltaKeyAngle = Quaternion.Angle(shinL.localRotation, shinLRef.localRotation) * Mathf.Deg2Rad;
        keyAngleReward += deltaKeyAngle * deltaKeyAngle;

        deltaKeyAngle = Quaternion.Angle(shinR.localRotation, shinRRef.localRotation) * Mathf.Deg2Rad;
        keyAngleReward += deltaKeyAngle * deltaKeyAngle;

        deltaKeyAngle = Quaternion.Angle(thighL.localRotation, thighLRef.localRotation) * Mathf.Deg2Rad;
        keyAngleReward += deltaKeyAngle * deltaKeyAngle;

        deltaKeyAngle = Quaternion.Angle(thighR.localRotation, thighRRef.localRotation) * Mathf.Deg2Rad;
        keyAngleReward += deltaKeyAngle * deltaKeyAngle;

        deltaKeyAngle = Quaternion.Angle(spine.localRotation, spineRef.localRotation) * Mathf.Deg2Rad;
        keyAngleReward += deltaKeyAngle * deltaKeyAngle;

        float outreward = (float)Math.Exp(-2 * keyAngleReward);

        return outreward;
    }
    /// <summary>
    /// Linear interpolation for 3D vectors
    /// </summary>
    /// <param name="comPosition"></param>
    /// <param name="begin"></param>
    /// <param name="iDataset"></param>
    /// <returns></returns>
    public Vector3 keyFrameLinearLerp(float[,] comPosition,int begin, float iDataset)
    {
        int index_ini = (int)Mathf.Floor(iDataset);
        float rate = iDataset - Mathf.Floor(iDataset);

        Vector3 com = new Vector3(Mathf.Lerp(comPosition[index_ini, begin], comPosition[index_ini + 1, begin], rate), Mathf.Lerp(comPosition[index_ini, begin + 1], comPosition[index_ini + 1, begin + 1], rate), Mathf.Lerp(comPosition[index_ini, begin + 2], comPosition[index_ini + 1, begin + 2], rate));
        return com;
    }

    /// <summary>
    /// Quaternion slerp for joint rotations
    /// </summary>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    public Quaternion keyFrameQLerp(float[,] walk, int begin, float iDataset)
    {
        int index_ini = (int)Mathf.Floor(iDataset);
        float rate = iDataset - Mathf.Floor(iDataset);
        Quaternion angle1 = new Quaternion(walk[index_ini, begin], walk[index_ini, begin + 1], walk[index_ini, begin + 2], walk[index_ini, begin + 3]);
        Quaternion angle2 = new Quaternion(walk[index_ini + 1, begin], walk[index_ini + 1, begin + 1], walk[index_ini + 1, begin + 2], walk[index_ini + 1, begin + 3]);

        Quaternion qlerp = Quaternion.Slerp(angle1, angle2, rate);
        return qlerp;
    }



    /// <summary>
    /// Range 0-1; equals 1 when no dragging occurs
    /// </summary>
    /// <returns></returns>
    private float FootDrag()
    {
        float rewardDrag = 0;

        float leftpre = m_JdController.bodyPartsDict[shinL].preContact ? 1 : 0;
        float rightpre = m_JdController.bodyPartsDict[shinR].preContact ? 1 : 0;
        float leftnow = m_JdController.bodyPartsDict[shinL].groundContact.touchingGround ? 1 : 0;
        float rightnow = m_JdController.bodyPartsDict[shinR].groundContact.touchingGround ? 1 : 0;

        float deltaLeft = Vector3.Distance(shinL.position, m_JdController.bodyPartsDict[shinL].PrePosition) / 0.2f;
        float deltaRight = Vector3.Distance(shinR.position, m_JdController.bodyPartsDict[shinR].PrePosition) / 0.2f;
        //Debug.Log("leftpre: "+ leftpre + " rightpre: " + rightpre + " leftnow: " + leftnow + " rightnow: " + rightnow);
        //Debug.Log("footLRef.position: " + footLRef.position + " [footLRef].PrePosition: " + m_JdController.bodyPartsDict[footLRef].PrePosition + " deltaLeft: " + Vector3.Distance(footLRef.position, m_JdController.bodyPartsDict[footLRef].PrePosition));
        //Debug.Log("footRRef.position: " + footRRef.position + " [footRRef].PrePosition: " + m_JdController.bodyPartsDict[footRRef].PrePosition + " deltaLeft: " + Vector3.Distance(footRRef.position, m_JdController.bodyPartsDict[footRRef].PrePosition));

        rewardDrag -= leftpre * leftnow * (1 - (float)Math.Exp(-deltaLeft * deltaLeft));
        rewardDrag -= rightpre * rightnow * (1 - (float)Math.Exp(-deltaRight * deltaRight));

        //Debug.Log("rewardleft: " + leftpre * leftnow * (1 - (float)Math.Exp(-deltaLeft * deltaLeft)) + " rewardright: " + rightpre * rightnow * (1 - (float)Math.Exp(-deltaRight * deltaRight)) );

        return rewardDrag;
    }

    /// <summary>
    /// Compute foot height above ground
    /// </summary>
    /// <param name="foot"></param>
    /// <returns></returns>
    public float FootHeight(Transform foot)
    {
        RaycastHit[] hit;// raycast collision detection

        List<float> rayDistancelist = new List<float>(); ;

        hit = Physics.RaycastAll(foot.position, Vector3.down, 500f, ~(1 << 0));// ray hits

        if (hit.Length == 0) { return 0; }
        else
        {
            for (int c = 0; c < hit.Length; c++)
            {
                if (hit[c].collider.gameObject.CompareTag("ground"))// collider is ground
                { rayDistancelist.Add(hit[c].distance); }
            }
            return rayDistancelist.Min();// nearest ground distance
        }
    }

    /// <summary>
    /// Record full-body posture: current rotation and position (world)
    /// </summary>
    public void RecordPostureFoot()
    {
        m_JdController.bodyPartsDict[shinL].preContact = m_JdController.bodyPartsDict[shinL].groundContact.touchingGround;
        m_JdController.bodyPartsDict[shinR].preContact = m_JdController.bodyPartsDict[shinR].groundContact.touchingGround;
        m_JdController.bodyPartsDict[shinLRef].preContact = m_JdController.bodyPartsDict[shinLRef].groundContact.touchingGround;
        m_JdController.bodyPartsDict[shinRRef].preContact = m_JdController.bodyPartsDict[shinRRef].groundContact.touchingGround;

        m_JdController.bodyPartsDict[shinL].PrePosition = shinL.position;
        m_JdController.bodyPartsDict[shinR].PrePosition = shinR.position;
        m_JdController.bodyPartsDict[shinLRef].PrePosition = shinLRef.position;
        m_JdController.bodyPartsDict[shinRRef].PrePosition = shinRRef.position;
    }

    /// <summary>
    /// Ground-contact visualization for objects tagged "mesh"
    /// </summary>
    public void GroundTouchVisualize()
    {

        foreach (var meshObject in meshChangeObject)
        {
            // find parent object
            Transform parentTransform = meshObject.transform.parent;
            // if parent has a body part entry
            if (m_JdController.bodyPartsDict.Keys.Contains(parentTransform))
            {
                meshObject.GetComponent<Renderer>().material = m_JdController.bodyPartsDict[parentTransform].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
                //Debug.Log(m_JdController.bodyPartsDict[parentTransform].groundContact.touchingGround);
            }
        }

    }
    public class ImuData2
    {
        /// <summary>
        /// Body positions and orientations for the used body segments
        /// </summary>
        public float[,] walk;
        /// <summary>
        /// End-effector positions; 0-2 footL, 3-5 footR, 6-8 spine
        /// </summary>
        public float[,] endPosition;
        /// <summary>
        /// Foot-ground contact; 0 footL, 1 footR; 1 means contact
        /// </summary>
        public int[,] footContact;
        public float[,] comPosition;
        /// <summary>
        /// Angular velocities for the used body segments
        /// </summary>
        public float[,] angularVelocity;
        public float[,] bodyPartVelocity;
        /// <summary>
        /// Walking speed over this clip
        /// </summary>
        public float aveVelocity;
        public float stepLength;
        public float stepFrequency;
        public float[,] fai;
    }

        /// <summary>
        /// Randomized pose initialization at episode start
        /// </summary>
    public void SetIniPosture(float keyini)
    {
        //Debug.Log("m_JdController.bodyPartsDict[hips].startingPos" + m_JdController.bodyPartsDict[hips].startingPos+ "xPositionBeforeI" + xPositionBeforeI);
        hips.position = m_JdController.bodyPartsDict[hips].startingPos + xPositionBeforeI + keyFrameLinearLerp(imuData2.walk, 0, keyini);

        var motionData = imuData2.walk;
        int count = 0;
        int countv = 0;
        int counta = 0;
        foreach (var trans in bodylistInUse)//q order: x y z w
        {
            // Set pose
            Quaternion bodyAngle = keyFrameQLerp(motionData, 3 + count * 4, keyini);
            if (trans == hips) trans.rotation = bodyAngle;
            else trans.localRotation = bodyAngle;
            count++;

            // Set rigidbody velocities
            m_JdController.bodyPartsDict[trans].rb.velocity = keyFrameLinearLerp(imuData2.bodyPartVelocity, countv * 3, keyini);
            m_JdController.bodyPartsDict[trans].rb.angularVelocity = keyFrameLinearLerp(imuData2.angularVelocity, counta * 3, keyini);
            countv++; counta++;
        }
    }
    /// <summary>
    /// Compute center of mass
    /// </summary>
    /// <returns>Center of mass in world space</returns>
    private Vector3 comCalculate()
    {
        Vector3 com = Vector3.zero; float mass = 0;
        List<Transform> bodylistInUse3 = new List<Transform> { hips, thighL, shinL, thighR, shinR, spine };
        //List<Transform> bodylistInUse3 = new List<Transform> { hipsRef, thighLRef, shinLRef, thighRRef, shinRRef, spineRef };

        foreach (var trans in bodylistInUse3)
        {
            com += m_JdController.bodyPartsDict[trans].rb.mass * trans.position;
            mass += m_JdController.bodyPartsDict[trans].rb.mass;
        }
        com /= mass;// center of mass
        return com;
    }
}

