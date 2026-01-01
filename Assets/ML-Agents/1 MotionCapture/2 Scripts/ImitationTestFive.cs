using System;
using Unity.MLAgentsExamples;
using UnityEngine;
//using UnityEditor;
using Newtonsoft.Json;
using System.IO;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;
using System.Linq;
using static WalkerRagdoll;
using Random = UnityEngine.Random;

// Imitation model: first mimics a chosen sequence from the dataset; after training, continues on a patient-specific sequence

public class ImitationTestFive : Agent
{
    private float framerate;
    //private float framerateDataset;
    public Transform Target;

    private float iniHipHeight;
    /// <summary>
    /// Objects that change color based on ground contact
    /// </summary>
    private GameObject[] meshChangeObject;
    public Material groundedMaterial;
    public Material unGroundedMaterial;

    /// <summary>
    /// Dictionary mapping step length to file name
    /// </summary>
    [HideInInspector] public Dictionary<float, string> fileStepLength;
    /// <summary>
    /// List of all JSON file names
    /// </summary>
    private List<string> fileList;

    /// <summary>
    /// Path of the current JSON file
    /// </summary>
    private string jsonDir;

    [Header("User Input Goal")]
    /// <summary>
    /// Randomize target step length (true during training, false during inference)
    /// </summary>
    public bool randomizeWalkLengthEachEpisode;
    /// <summary>
    /// Randomize target speed (true during training, false during inference)
    /// </summary>
    public bool randomizeWalkSpeedEachEpisode;
    /// <summary>
    /// User-input target step length for two steps (~0.7-2, dataset dependent)
    /// </summary>
    public float lengthInput;
    /// <summary>
    /// User-input target speed (expected 1-2)
    /// </summary>
    public float velocityInput;
    /// <summary>
    /// Enable to use patient-specific data for personalization
    /// </summary>
    public bool usePatientData;

    public int velocityRange; // number of samples to compute average speed (default 100)

    [Header("Training Data !! Don't Edit !!")]
    /// <summary>
    /// Phase parameter in [0,1]
    /// </summary>
    public float fai;
    /// <summary>
    /// Counter for fai; resets each cycle
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
    /// Dataset index for reward interpolation (float for lerp)
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

    private JointDriveController m_JdController;
    /// <summary>
    /// Joints used in the simulation; upper limbs excluded for this experiment
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
        fileList = GetFiles(Application.streamingAssetsPath + "/OutputHealthDataset/", ".json");

        meshChangeObject = GameObject.FindGameObjectsWithTag("mesh");
        framerate = 30;

        bodylistInUse = new List<Transform> { hips, thighL, shinL, thighR, shinR, spine };

        // Set this before imitating a specific patient
        dirPatient = Application.streamingAssetsPath+"/OutputPatientDataset/walk475.json";

        if (usePatientData)
        {
            randomizeWalkLengthEachEpisode = false;
            randomizeWalkSpeedEachEpisode = false;
            imuData2 = JsonConvert.DeserializeObject<ImuData2>(File.ReadAllText(@dirPatient));
            lengthInput = imuData2.stepLength;
            velocityInput = imuData2.aveVelocity;
        }

    }

    public override void Initialize()
    {
        Time.fixedDeltaTime = 1.0f / framerate;// physics step duration

        // Setup each body part
        m_JdController = GetComponent<JointDriveController>();
        m_JdController.SetupBodyPart(hips);
        m_JdController.SetupBodyPart(spine);
        m_JdController.SetupBodyPart(thighL);
        m_JdController.SetupBodyPart(shinL);
        m_JdController.SetupBodyPart(thighR);
        m_JdController.SetupBodyPart(shinR);

        // Reference joints (toggle if needed)
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

        fileStepLength = new Dictionary<float, string>();
        LogStepLengthInJson(fileList);// log step length for each file
        iniHipHeight = FootHeight(hips);

        showtext = true;
    }



    public override void OnEpisodeBegin()
    {
        i = 0;
        showtext = true;
        // Initialize agent pose
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values) { bodyPart.Reset(bodyPart); };

        if (usePatientData) jsonDir = dirPatient;// use provided patient file
        else jsonDir = FindFileByStepLength(lengthGoal);// select sequence based on target step length

        // Select reference clip
        lengthGoal =
            randomizeWalkLengthEachEpisode ? Random.Range(0.5f, 1.7f) : lengthInput;// example fixed: 1.5096f
        velocityGoal =
            randomizeWalkSpeedEachEpisode ? Random.Range(0.4f, 1.4f) : velocityInput;// example fixed: 1.3828f
        freGoal = velocityGoal / lengthGoal;

        //Debug.Log("jsonDir: " + jsonDir + " lengthGoal: " + lengthGoal + " velocityGoal: " + velocityGoal);
        imuData2 = JsonConvert.DeserializeObject<ImuData2>(File.ReadAllText(@jsonDir));// reference for this episode

        // Randomly choose a frame as initial pose; compute clip lengths and starting indices
        clipLengthDataset = imuData2.walk.Length / 27;
        clipLengthImitation = Mathf.RoundToInt(framerate / freGoal);
        Debug.Log("clipLengthDataset: " + clipLengthDataset + " clipLengthImitation: " + clipLengthImitation+ " jsonDir: " + jsonDir);


        i = Random.Range(0, clipLengthImitation-1);// position within imitation clip
        //i = 0;// test option
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



    public override void CollectObservations(VectorSensor sensor)
    {
        if(positionlist.Count>1) showtext = false;// only show on first few steps
        GroundTouchVisualize();// change color on ground contact

        // Phase parameter
        sensor.AddObservation(fai);
        //Debug.Log("fai: "+ fai + "lengthGoal"+ (lengthGoal - 1.1f) / 0.6f + " FootHeight(hips): " + (FootHeight(hips) - 0.2f) / 0.8f);
        // Target step length (normalized)
        sensor.AddObservation((lengthGoal-0.5f)/1.2f);
        // Step length ratio
        if (comPositionlist.Count > clipLengthImitation)
            currentLength = comPositionlist[comPositionlist.Count - 1].x - comPositionlist[comPositionlist.Count - 1 - clipLengthImitation].x;
        else if (comPositionlist.Count == 0)
            currentLength = 0;
        else
            currentLength = comPositionlist[comPositionlist.Count - 1].x - comPositionlist[0].x;
        sensor.AddObservation(currentLength / lengthGoal);
        //Debug.Log("Current frames: " + comPositionlist.Count + " currentLength: " + currentLength + " lengthGoal: " + lengthGoal + " Observation:" + (currentLength/ lengthGoal));

        // Hip height above ground (normalized)
        sensor.AddObservation((FootHeight(hips)-0.2f)/0.8f);
        //Debug.Log("FootHeight(hips)"+ FootHeight(hipsRef));
        // Velocity error
        int init;
        init = positionlist.Count > velocityRange ? positionlist.Count - velocityRange : 0;  // starting index for averaging
        averVelocity = positionlist.Count>1?Mathf.Abs(positionlist[positionlist.Count-1] - positionlist[init]) / (1.0f / framerate * (positionlist.Count- init-1)): m_JdController.bodyPartsDict[hips].rb.velocity.x;
        averComVelocity = positionlist.Count > 1 ? (comPositionlist[positionlist.Count - 1] - comPositionlist[init]) / (1.0f / framerate * (positionlist.Count - init - 1)) : m_JdController.bodyPartsDict[hips].rb.velocity;
        averComVelocity = new Vector3(averComVelocity.x,0, averComVelocity.z);

        sensor.AddObservation(averComVelocity.z/ velocityGoal + 0.5f);// centered at 0.5, lateral sway
        //sensor.AddObservation(Vector3.one);
        sensor.AddObservation((velocityGoal - Vector3.Dot(averComVelocity, Vector3.right)) / velocityGoal / 2 / 1.5f + 0.5f);// centered at 0.5

        // Target speed (normalized)
        sensor.AddObservation((velocityGoal-0.4f)/1.0f);
        //Debug.Log("velocityGoal: " + (velocityGoal-0.3f)/1.7f + " speed delta 1: " + ((velocityGoal - Vector3.Dot(averComVelocity, Vector3.right)) / velocityGoal / 2/1.5f + 0.5f) + " lateral speed delta 2: " + (averComVelocity.z / velocityGoal + 0.5f));

        // Ground contact flags
        sensor.AddObservation(m_JdController.bodyPartsDict[shinL].groundContact.touchingGround);
        sensor.AddObservation(m_JdController.bodyPartsDict[shinR].groundContact.touchingGround);

        // Joint information

        // Hips: rotation + velocities
        sensor.AddObservation(hips.rotation);
        // Linear velocity (world) normalized
        sensor.AddObservation(m_JdController.bodyPartsDict[hips].rb.velocity / 8f + 0.5f * Vector3.one);
        // Angular velocity (world) normalized
        sensor.AddObservation(m_JdController.bodyPartsDict[hips].rb.angularVelocity / 30f + 0.5f * Vector3.one);
        // Root position intentionally omitted
        //Debug.Log("m_JdController.bodyPartsDict[hips].rb.velocity: " + (m_JdController.bodyPartsDict[hips].rb.velocity / 8f + 0.5f * Vector3.one) + " m_JdController.bodyPartsDict[hips].rb.angularVelocity: " + (m_JdController.bodyPartsDict[hips].rb.angularVelocity / 30f + 0.5f * Vector3.one));


        //List<Transform> bodylistInUse = new List<Transform> { hips, thighL, shinL, thighR, shinR, spine };
        foreach (var trans in bodylistInUse)
        {
            Unity.MLAgentsExamples.BodyPart bp = m_JdController.bodyPartsDict[trans];

            if (trans != hips)
            {
                // Local rotation
                sensor.AddObservation(trans.localRotation);
                // Linear velocity (world) normalized
                sensor.AddObservation(bp.rb.velocity / 8f + 0.5f * Vector3.one);
                // Angular velocity (world) normalized
                sensor.AddObservation(bp.rb.angularVelocity / 30f + 0.5f * Vector3.one);
                // Torque limit ratio
                sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
                //Debug.Log("bp.currentStrength / m_JdController.maxJointForceLimit: " + bp.currentStrength / m_JdController.maxJointForceLimit );
                //Debug.Log("m_JdController.bodyPartsDict[transs].rb.velocity: " + (bp.rb.velocity / 8f + 0.5f * Vector3.one) + " m_JdController.bodyPartsDict[trans].rb.angularVelocity: " + (bp.rb.angularVelocity / 30f + 0.5f * Vector3.one));

            }
            if (trans == shinL || trans == shinR || trans == spine)// end-effector positions relative to hips
                sensor.AddObservation((trans.position - hips.position) / 1.5f + 0.5f * Vector3.one);
            //Debug.Log("trans.position - hips.position: " + ((trans.position - hips.position) / 1.5f + 0.5f * Vector3.one));
        }
    }

    /// <summary>
    /// Apply actions to joints
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

        if (Math.Abs(hips.position.x - Target.position.x) < 1)// stop when close to target
        {
            SetReward(1);
            Debug.Log("Reached target");
            EndEpisode();
        }

        // Add rewards; currently unified path
        TotalRewardCalculate(imuData2, iDataset);

        RecordPostureFoot();

        i++;// update after computing reward
        if (i == clipLengthImitation) i = 0; // wrap back to fai=0
        fai = (float)i / clipLengthImitation;
        iDataset = fai * (clipLengthDataset - 1) > (clipLengthDataset - 1) ? (clipLengthDataset - 1) : fai * (clipLengthDataset - 1);

        MotionDataControlRef(imuData2, iDataset);
    }


    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;

            for (int n = 0; n < continuousActions.Length; n++)
            {
                continuousActions[n] = Random.Range(-1f, 1f);
            }
    }
    private void MotionDataControlRef(ImuData2 imuData2, float iDataset)// 35 dimensions total
    {
        // Reuse this to hold intermediate position
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

    /// <summary>
    /// Compute reward using the reference dataset
    /// </summary>
    public void TotalRewardCalculate(ImuData2 imuData2, float iDataset)
    {
        // 1: Task completion (terrain, direction, speed)
        // 1.1 Hip height matches reference; add lateral consistency
        var delatHeight = (Mathf.Abs(FootHeight(hips) - FootHeight(hipsRef)));// hip-to-ground height should be ~0.7-0.9
        var delatLateral = (hips.position.z-m_JdController.bodyPartsDict[hips].startingPos.z)-(hipsRef.position.z - m_JdController.bodyPartsDict[hipsRef].startingPos.z);
        float hipHeightReward = 0.5f*(float)Math.Exp(-40 * delatHeight * delatHeight) + 0.5f * (float)Math.Exp(-40 * delatLateral * delatLateral);
        hipHeightReward = (float)Math.Exp(-40 * delatHeight * delatHeight) ;// trying only this term

        // 1.2 Walk generally forward (+x); omit full axis comparison for now
        Vector3 direction = new Vector3(hips.position.x - m_JdController.bodyPartsDict[hips].startingPos.x, 0, hips.position.z - m_JdController.bodyPartsDict[hips].startingPos.z).normalized;
        float deltadirection = 1 - Mathf.Clamp( Vector3.Dot(direction.normalized, Vector3.right),0,1);
        float forwardReward = (float)Math.Exp(-deltadirection * deltadirection);

        // 1.3 Match target speed
        float deltaV = velocityGoal - Vector3.Dot(averComVelocity, Vector3.right);
        float velocityReward = Mathf.Exp(-2.5f * deltaV * deltaV);

        // 1.4 Match step length (approx)
        float deltaLength = 1 - Mathf.Clamp(currentLength / lengthGoal, 0, 1);
        float LengthReward = (float)Math.Exp(-deltaLength * deltaLength);
        //Debug.Log(" data length: " + deltaLength + " currentLength: " + currentLength+" imitation length: " + lengthGoal + " LengthReward: " + LengthReward);// verified

        //float goalReward = hipHeightReward * velocityReward * forwardReward * LengthReward*10;
        float goalReward = velocityReward*0.5f + forwardReward*0.2f+ hipHeightReward *0.2f+ LengthReward*0.1f;// lateral constraint can be relaxed
        //Debug.Log("velocityReward: " + velocityReward + " hipHeightReward: " + hipHeightReward + " goalReward: " + goalReward);

        // 2: Gait quality (grounding, drag, distance)
        // 2.1 Both shins grounded simultaneously is discouraged
        float bodyStableReward = 0;
        if (m_JdController.bodyPartsDict[shinL].groundContact.touchingGround && !m_JdController.bodyPartsDict[shinR].groundContact.touchingGround)
        { bodyStableReward = -0.005f; }
        // 2.2 No dragging
        float footDragPenal = FootDrag();// zero when airborne, negative when dragging
        // 2.3 Encourage progress
        float walklengthReward = 0.005f;

        float walkAbilityReward = footDragPenal;
        //Debug.Log("bodyStableReward(not grounded): " + bodyStableReward + " footDragPenal: " + footDragPenal + " walklengthReward: " + walklengthReward);


        // 3: Imitation similarity
        // Target frame index: iDataset

        // 3.1 COM and end-effector positions relative to hips
        float delta1 = Vector3.Magnitude(comCalculate() - hips.position - keyFrameLinearLerp(imuData2.comPosition, 0, iDataset));// expected ~0.07-0.1
        float delta2 = Vector3.Magnitude(shinL.position - hips.position - keyFrameLinearLerp(imuData2.endPosition, 0, iDataset));// expected ~0.07
        float delta3 = Vector3.Magnitude(shinR.position - hips.position - keyFrameLinearLerp(imuData2.endPosition, 3, iDataset));
        float delta4 = Vector3.Magnitude(spine.position - hips.position - keyFrameLinearLerp(imuData2.endPosition, 6, iDataset));// expected ~0.01
        float keypositionReward = (float)Math.Exp(-15 * delta2 * delta2 -15* delta3 * delta3 - 30 * delta4 * delta4);
        float compositionReward = (float)Math.Exp(-50 * delta1 * delta1);

        //Debug.Log("delta1: "+ delta1 + " delta2: " + delta2 + " delta3: " + delta3 + " delta4: " + delta4);
        //Debug.Log("Math.Exp delta1: " + Math.Exp(-50*delta1 * delta1) + " delta2: " + Math.Exp(-10 * delta2 * delta2) + " delta3: " + Math.Exp(-10 * delta3 * delta3) + " delta4: " + Math.Exp(-10 * delta4 * delta4));
        //Debug.Log(" compositionReward: " + compositionReward + "keypositionReward: " + keypositionReward );


        // 3.2 Joint angles, angular velocity, and linear velocity
        //bodylistInUse = new List<Transform> { hips, thighL, shinL, thighR, shinR, spine };
        //List<Transform> bodylistInUse4 = new List<Transform> { hipsRef, thighLRef, shinLRef, thighRRef, shinRRef, spineRef };
        float keyAngleReward = 0;
        float keyAngleVelociyReward = 0;
        float keyVelocityReward = 0;
        int count = 0;
        foreach (var trans in bodylistInUse)
        {
            Unity.MLAgentsExamples.BodyPart bp = m_JdController.bodyPartsDict[trans];

            // Joint linear velocity in world frame
            float deltaKeyV = Vector3.Magnitude(bp.rb.velocity - keyFrameLinearLerp(imuData2.bodyPartVelocity, count*3, iDataset));
            if (trans != shinL && trans != shinL)
                keyVelocityReward += deltaKeyV * deltaKeyV;
            else
                keyVelocityReward += 0.5f * deltaKeyV * deltaKeyV;

            // Joint angular velocity in world frame
            float deltaKeyAngleV = Vector3.Magnitude(bp.rb.angularVelocity - keyFrameLinearLerp(imuData2.angularVelocity, count * 3, iDataset));
            if (trans != shinL && trans != shinL)
                keyAngleVelociyReward += deltaKeyAngleV * deltaKeyAngleV;
            else
                keyAngleVelociyReward += 0.5f * deltaKeyAngleV * deltaKeyAngleV;

            // Joint angle (quaternion)
            Quaternion bodyAngle = trans == hips ? trans.rotation : trans.localRotation;
            Quaternion datasetAngle = keyFrameQLerp(imuData2.walk, 3 + count * 4, iDataset);
            float deltaKeyAngle = Quaternion.Angle(bodyAngle, datasetAngle) *Mathf.Deg2Rad;
            if (trans != shinL && trans != shinL)
                keyAngleReward += deltaKeyAngle * deltaKeyAngle;
            else
                keyAngleReward += 0.5f*deltaKeyAngle * deltaKeyAngle;

            count++;
        }

        keyVelocityReward = (float)Math.Exp(-keyVelocityReward);// simplified term
        keyAngleVelociyReward = (float)Math.Exp(-0.1f * keyAngleVelociyReward);
        //Debug.Log("keyAngleReward from dataset: " + (float)Math.Exp(-2 * keyAngleReward)+ " BodyAngleRewardinLocalAxis(): "+ BodyAngleRewardinLocalAxis() + " BodyAngleRewardinWorldAxis(): " + BodyAngleRewardinWorldAxis());
        keyAngleReward = BodyAngleRewardinLocalAxis(); // compare in local frame; BodyAngleRewardinWorldAxis();(float)Math.Exp(-2 * keyAngleReward);

        float keyFrameReward = 0.55f * keyAngleReward + 0.15f * keyAngleVelociyReward + 0.05f * keyVelocityReward + 0.15f * keypositionReward + 0.1f * compositionReward;

        float totalReward = 0.5f * goalReward + 0.5f * keyFrameReward + walkAbilityReward;
        totalReward = (0.5f * goalReward + 0.5f * keyFrameReward + walkAbilityReward) * 2 - 1;
        //totalReward = keyAngleReward*0.2f + walklengthReward + hipHeightReward*0.5f + LengthReward*0.3f;// debug variant
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
    }


    /// <summary>
    /// Joint-angle reward computed in world space
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
        keyAngleReward += 0.5f * deltaKeyAngle * deltaKeyAngle;

        deltaKeyAngle = Quaternion.Angle(shinR.rotation, shinRRef.rotation) * Mathf.Deg2Rad;
        keyAngleReward += 0.5f * deltaKeyAngle * deltaKeyAngle;

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
    /// Joint-angle reward computed in local space
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
        keyAngleReward += 0.5f*deltaKeyAngle * deltaKeyAngle;

        deltaKeyAngle = Quaternion.Angle(shinR.localRotation, shinRRef.localRotation) * Mathf.Deg2Rad;
        keyAngleReward += 0.5f*deltaKeyAngle * deltaKeyAngle;

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
    /// Linear interpolation over a 3D vector sequence
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
    /// Quaternion interpolation between keyframes
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
    /// Range 0-1; 1 when no dragging
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
        RaycastHit[] hit;// raycast hits

        List<float> rayDistancelist = new List<float>(); ;

        hit = Physics.RaycastAll(foot.position, Vector3.down, 500f, ~(1 << 0));// with collisions

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
    /// Compute angular velocity from quaternion change (rad/s)
    /// </summary>
    /// <param name="lastRoation"></param>
    /// <param name="thisRotation"></param>
    /// <returns></returns>
    public Vector3 AngularVelocityCalculate(Quaternion PreRoation, Quaternion thisRotation)
    {
        float angleInDegrees;
        Vector3 rotationAxis;
        Quaternion myQuaternion = Quaternion.Inverse(PreRoation) * thisRotation;

        myQuaternion.ToAngleAxis(out angleInDegrees, out rotationAxis);

        Vector3 angularDisplacement = rotationAxis * angleInDegrees * Mathf.Deg2Rad;
        Vector3 AngularVelocityThis = angularDisplacement / Time.fixedDeltaTime;

        return AngularVelocityThis;
    }


    /// <summary>
    /// Record posture for shins, including contact and positions
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
    /// For all objects tagged "mesh", detect ground contact and change color
    /// </summary>
    public void GroundTouchVisualize()
    {

        foreach (var meshObject in meshChangeObject)
        {
            // Find parent transform
            Transform parentTransform = meshObject.transform.parent;
            // Only handle meshes whose parent is a controlled body part
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
        /// Body pose for required segments
        /// </summary>
        public float[,] walk;
        /// <summary>
        /// End-effector positions: 0-2 footL, 3-5 footR, 6-8 spine
        /// </summary>
        public float[,] endPosition;
        /// <summary>
        /// Foot-ground contact: 0 footL, 1 footR; 1=contact, 0=air
        /// </summary>
        public int[,] footContact;
        public float[,] comPosition;
        /// <summary>
        /// Angular velocity for required segments
        /// </summary>
        public float[,] angularVelocity;
        public float[,] bodyPartVelocity;
        /// <summary>
        /// Step speed during the clip
        /// </summary>
        public float aveVelocity;
        public float stepLength;
        public float stepFrequency;
        public float[,] fai;
    }

    /// <summary>
    /// Populate fileStepLength dictionary
    /// </summary>
    public void LogStepLengthInJson(List<string> fileList)
    {
        string Dir;
        ImuData2 imuData0;

        for (int y = 0; y < fileList.Count; y++)
        {
            Dir = fileList[y];
            imuData0 = JsonConvert.DeserializeObject<ImuData2>(File.ReadAllText(@Dir));
            fileStepLength.Add(imuData0.stepLength, Dir);
        }
    }

    /// <summary>
    /// Choose the sequence closest to target step length
    /// </summary>
    /// <param name="length"></param>
    /// <returns></returns>
    public string FindFileByStepLength(float steplength)
    {
        string jsonDir;
        float[] lengthList_origin = new float[fileStepLength.Count];
        float[] lengthList_abs = new float[fileStepLength.Count];
        int y = 0;
        foreach (var key in fileStepLength.Keys)
        {
            lengthList_origin[y] = key;
            lengthList_abs[y] = Math.Abs(key - steplength);
            y++;
            //Debug.Log("key"+ key+ " key - length" + Math.Abs(key - steplength));
        }
        var (minValue, minIndex) = lengthList_abs.Select((x, i) => (x, i)).Min();// find frame with closest step length

        //Debug.Log("Closest file to target step length minValue " + minValue + " minIndex " + minIndex + " lengthList_origin[minIndex] "+ lengthList_origin[minIndex]);
        jsonDir = fileStepLength[lengthList_origin[minIndex]];
        //Debug.Log("Closest file to target step length minValue " + minValue + " minIndex" + minIndex + " jsonDir " + jsonDir);
        return jsonDir;
    }

        /// <summary>
    /// Randomize initial pose at frame start
    /// </summary>
    public void SetIniPosture(float keyini)
    {
        //Debug.Log("m_JdController.bodyPartsDict[hips].startingPos" + m_JdController.bodyPartsDict[hips].startingPos+ "xPositionBeforeI" + xPositionBeforeI);
        hips.position = m_JdController.bodyPartsDict[hips].startingPos + xPositionBeforeI + keyFrameLinearLerp(imuData2.walk, 0, keyini);

        var motionData = imuData2.walk;
        int count = 0;
        int countv = 0;
        int counta = 0;
        foreach (var trans in bodylistInUse)// q order: x y z w
        {
            // Apply pose from reference
            Quaternion bodyAngle = keyFrameQLerp(motionData, 3 + count * 4, keyini);
            if (trans == hips) trans.rotation = bodyAngle;
            else trans.localRotation = bodyAngle;
            count++;
            // Set rigidbody velocities
            m_JdController.bodyPartsDict[trans].rb.velocity = keyFrameLinearLerp(imuData2.bodyPartVelocity, countv*3, keyini);
            m_JdController.bodyPartsDict[trans].rb.angularVelocity = keyFrameLinearLerp(imuData2.angularVelocity, counta*3, keyini);
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


