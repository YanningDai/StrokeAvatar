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

/// Motion prediction model: supports demo recording and testing (RL + BC + GAIL)
public class StrokeWalker : Agent
{

    private float framerate = 30;
    /// <summary>
    /// When true, load observation/action from json and record demo (stay in place; used in heuristic mode)
    /// </summary>
    public bool flagDemo;
    public Transform Target;
    [Header("Body Parts")]
    public Transform hips;
    public Transform chest;
    public Transform spine;
    public Transform head;
    public Transform thighL;
    public Transform shinL;
    public Transform footL;
    public Transform thighR;
    public Transform shinR;
    public Transform footR;
    public Transform armL;
    public Transform forearmL;
    public Transform handL;
    public Transform armR;
    public Transform forearmR;
    public Transform handR;

    private JointDriveController m_JdController;
    public StatsRecorder statsRecorder;// tensorboard output
    private string dirOut;
    private JsonDataLog jsonDataLog;

    // logs from inference after training
    private List<List<float>> observationLogList;
    private List<List<float>> actionLogList;
    private int jsonN;

    private float IniFootHeightLeft;
    private float IniFootHeightRight;
    private float IniHipHeight;
    private float averVelocityHip;

    /// <summary>
    /// Target speed input by user (around (1-2,0,0))
    /// </summary>
    public float targetVelocity;
    private float velGoal;// actual target speed
    /// <summary>
    /// Randomize target speed; true during training, false during inference
    /// </summary>
    public bool randomizeWalkSpeedEachEpisode;

    private int step;// step index within episode
    public int velocityRange; // number of frames used for average speed (default 100)
    private List<float> positionlist;
    float averVelocity;

    float dataV;// target speed from last demo frame
    int keyini;int keyend;
    int KeyPhaseNum;// current phase index (e.g., 1-27)
    int KeyPhaseTotal;// number of phases used (e.g., 27; dataset has 37)
    int KeyPhase;// current dataset frame index (e.g., 30)

    protected override void Awake()
    {
        base.Awake();
        statsRecorder = Academy.Instance.StatsRecorder;// tensorboard output
        dirOut = Path.Combine(Application.dataPath, "ObserAction/walk73.json");
        Application.targetFrameRate = 30;

        keyini = 6;keyend = 40;

    }

    public override void Initialize()
    {
        Time.fixedDeltaTime = 1.0f / framerate;// physics fixed timestep

        //Setup each body part
        m_JdController = GetComponent<JointDriveController>();
        m_JdController.SetupBodyPart(hips);
        m_JdController.SetupBodyPart(chest);
        m_JdController.SetupBodyPart(spine);
        m_JdController.SetupBodyPart(head);
        m_JdController.SetupBodyPart(thighL);
        m_JdController.SetupBodyPart(shinL);
        m_JdController.SetupBodyPart(footL);
        m_JdController.SetupBodyPart(thighR);
        m_JdController.SetupBodyPart(shinR);
        m_JdController.SetupBodyPart(footR);
        m_JdController.SetupBodyPart(armL);
        m_JdController.SetupBodyPart(forearmL);
        m_JdController.SetupBodyPart(handL);
        m_JdController.SetupBodyPart(armR);
        m_JdController.SetupBodyPart(forearmR);
        m_JdController.SetupBodyPart(handR);

        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        // Read json file
        jsonDataLog = JsonConvert.DeserializeObject<JsonDataLog>(File.ReadAllText(@dirOut));
        observationLogList = jsonDataLog.observationLog;// stored at 30 Hz
        actionLogList = jsonDataLog.actionLog;
        dataV = observationLogList[observationLogList.Count - 1][4];// target speed (fixed for this demo)

    }


    public override void OnEpisodeBegin()
    {
        // Initialize pose
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        // Resolve target speed
        velGoal = randomizeWalkSpeedEachEpisode ? UnityEngine.Random.Range(0.3f, 2) : targetVelocity;
        velGoal = 1.5f;
        IniFootHeightLeft = FootHeight(footL);
        IniFootHeightRight = FootHeight(footR);
        IniHipHeight = FootHeight(hips);
        jsonN = 0;

        RecordPosture();// record current rotation/position (world space)

        // Initialize speed tracking
        positionlist = new List<float>();
        positionlist.Add(m_JdController.bodyPartsDict[hips].startingPos.x);
        averVelocity = 0;
        step = 1;
        KeyPhaseNum = 0;
        KeyPhaseTotal = (int)Math.Round(dataV / targetVelocity * (keyend - keyini + 1), 0);// dataset has 37 frames for this speed
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        positionlist.Add(hips.position.x);

        if (flagDemo)// record demo
        {
            int n = 0;
            Vector3 vectorTem; Quaternion qTem; float floatTem; bool boolTem;

            //(hip height above ground - initial)/max height, float, 1
            floatTem = observationLogList[jsonN][n++]; sensor.AddObservation(floatTem);
            //Debug.Log("(hip height delta)/max " + floatTem);

            //(foot height above ground - initial)/max height, float, 2
            floatTem = observationLogList[jsonN][n++]; sensor.AddObservation(floatTem);
            floatTem = observationLogList[jsonN][n++]; sensor.AddObservation(floatTem);
            //Debug.Log("(foot height delta)/max " + floatTem);

            // target speed - average speed over window, float, 1
            floatTem = observationLogList[jsonN][n++]; sensor.AddObservation(floatTem);
            //Debug.Log("(target speed - avg)/target speed " + floatTem);

            // target speed, float, 1
            floatTem = observationLogList[jsonN][n++]; sensor.AddObservation(floatTem);

            // body forward vs target direction diff, quaternion, 4
            qTem = new Quaternion(observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++]);
            sensor.AddObservation(qTem);
            //Debug.Log("Body forward vs target diff " + qTem);

            // foot-ground contact, bool, 2
            floatTem = observationLogList[jsonN][n++]; if (floatTem == 1) { boolTem = true; } else { boolTem = false; }
            sensor.AddObservation(boolTem);
            //Debug.Log("Left foot " + boolTem);
            floatTem = observationLogList[jsonN][n++]; if (floatTem == 1) { boolTem = true; } else { boolTem = false; }
            sensor.AddObservation(boolTem);
            //Debug.Log("Right foot " + boolTem);

            // Add joint info
            // hips first: 4+3+3=10

            // global rotation quaternion, 4
            qTem = new Quaternion(observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++]);
            sensor.AddObservation(qTem);
            //Debug.Log("Hip rotation " + qTem);

            // linear velocity, Vector3, 3
            vectorTem = new Vector3(observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++]);
            sensor.AddObservation(vectorTem);
            //Debug.Log("Linear velocity " + vectorTem);

            // angular velocity, Vector3, 3
            vectorTem = new Vector3(observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++]);
            sensor.AddObservation(vectorTem);
            //Debug.Log("Angular velocity " + vectorTem);

            for (int m = 0; m < 8; m++)
            {
                // local rotation quaternion, 4
                qTem = new Quaternion(observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++]);
                sensor.AddObservation(qTem);
                //Debug.Log("local rotation quaternion " + qTem);

                // linear velocity, 3
                vectorTem = new Vector3(observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++]);
                sensor.AddObservation(vectorTem);
                //Debug.Log("Linear velocity " + vectorTem);

                // angular velocity, 3
                vectorTem = new Vector3(observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++]);
                sensor.AddObservation(vectorTem);
                //Debug.Log("Angular velocity " + vectorTem);

                // torque limit, 1
                floatTem = observationLogList[jsonN][n++]; sensor.AddObservation(floatTem);
                //Debug.Log("Torque limit " + floatTem);
            }
            // Because we skip SetPosture, reset each frame to avoid falling
            foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
            { bodyPart.Reset(bodyPart); }
            // total 109 values
        }

        else// normal run, total 109
        {
            JointVelocityCalculate();

            //(hip height above ground - initial) / max height, float, 1
            sensor.AddObservation((FootHeight(hips) - IniHipHeight) / 0.01f);

            //(foot height above ground - initial)/max height, float, 2
            sensor.AddObservation(Math.Max(0, FootHeight(footL) - IniFootHeightLeft) / 0.3f);
            sensor.AddObservation(Math.Max(0, FootHeight(footR) - IniFootHeightRight) / 0.3f);

            // target speed - average speed over window, float, 1
            int init;
            if (step > velocityRange) { init = step - velocityRange; } else { init = 0; }// init is window start
            averVelocityHip = (hips.position.x - positionlist[init]) / (1.0f / framerate * (step - init));

            averVelocity = 0;int q = 0;
            foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
            {
                averVelocity += bodyPart.rb.velocity.x;
                q++;
            }
            averVelocity /= q;

            sensor.AddObservation(velGoal - averVelocityHip);

            // target speed, float, 1
            sensor.AddObservation(velGoal);

            // body forward vs target direction diff, quaternion, 4
            Vector3 bodyForward = (spine.forward + chest.forward) / 2;
            Vector3 bodyForwardGoal = Vector3.right;
            sensor.AddObservation(Quaternion.FromToRotation(bodyForward, bodyForwardGoal));

            // foot-ground contact, bool, 2
            sensor.AddObservation(m_JdController.bodyPartsDict[footL].groundContact.touchingGround);
            sensor.AddObservation(m_JdController.bodyPartsDict[footR].groundContact.touchingGround);

            // Append per-joint info
            var bpDict = m_JdController.bodyPartsDict;
            List<BodyPart> bpList = new List<BodyPart>() { bpDict[chest], bpDict[spine], bpDict[thighL], bpDict[shinL], bpDict[footL], bpDict[thighR], bpDict[shinR], bpDict[footR] };

            // hips first: 4+3+3=10
            // global rotation quaternion, 4
            sensor.AddObservation(hips.rotation);
            // linear velocity, Vector3, 3
            sensor.AddObservation(bpDict[hips].velocity);
            // angular velocity, Vector3, 3
            sensor.AddObservation(bpDict[hips].angularVelocity);

            foreach (var bp in bpList)
            {// total 8 joints, each has 4+3+3+1=11
                Transform trans = bp.rb.GetComponent<Transform>();
                // local rotation, quaternion, 4
                sensor.AddObservation(trans.localRotation);
                // linear velocity, Vector3, 3
                sensor.AddObservation(bp.velocity);
                // angular velocity, Vector3, 3
                sensor.AddObservation(bp.angularVelocity);
                // torque limit, float, 1
                sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);

            }
        }
    }

    /// <summary>
    /// Action function (shared across scripts)
    /// </summary>
    /// <param name="actions"></param>
    public override void OnActionReceived(ActionBuffers actions)
    {
        var bpDict = m_JdController.bodyPartsDict;
        var n = -1;

        var continuousActions = actions.ContinuousActions;
        bpDict[chest].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], continuousActions[++n]);
        bpDict[spine].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], continuousActions[++n]);

        bpDict[thighL].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], 0);
        bpDict[thighR].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], 0);
        bpDict[shinL].SetJointTargetRotation(continuousActions[++n], 0, 0);
        bpDict[shinR].SetJointTargetRotation(continuousActions[++n], 0, 0);
        bpDict[footR].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], continuousActions[++n]);
        bpDict[footL].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], continuousActions[++n]);

        //update joint strength settings
        bpDict[chest].SetJointStrength(continuousActions[++n]);
        bpDict[spine].SetJointStrength(continuousActions[++n]);
        bpDict[thighL].SetJointStrength(continuousActions[++n]);
        bpDict[shinL].SetJointStrength(continuousActions[++n]);
        bpDict[footL].SetJointStrength(continuousActions[++n]);
        bpDict[thighR].SetJointStrength(continuousActions[++n]);
        bpDict[shinR].SetJointStrength(continuousActions[++n]);
        bpDict[footR].SetJointStrength(continuousActions[++n]);

        TotalRewardCalculate();

        // Record state at end
        RecordPosture();


        if (Math.Abs(hips.position.x - Target.position.x) < 1)// stop near target
        {
            SetReward(1);
            Debug.Log("Reached target");
            EndEpisode();
        }
        if (hips.position.y < bpDict[hips].startingPos.y - 2)// stop if fallen
        {
            SetReward(-1);
            Debug.Log("Fell off platform");
            EndEpisode(); }
        step++;
        
    }


    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        if (flagDemo)// fill actions from demo
        {
            for (int n = 0; n < continuousActions.Length; n++)
            {
                continuousActions[n] = actionLogList[jsonN][n];
            }
            jsonN++;

            if (jsonN == actionLogList.Count)
            {
                Debug.Log("Demo recording complete "+ jsonN);
                jsonN = 0;
                // Recording cannot call EndEpisode here
            }
        }
        else
        {
            for (int n = 0; n < continuousActions.Length; n++)
            {
                continuousActions[n] = 0;
            }
        }
    }

    /// <summary>
    /// Reward calculation
    /// </summary>
    public void TotalRewardCalculate()
    {
        float totalReward;
        float goalReward;
        //float walkAbilityReward;
        float keyFrameReward;

        float hipHeightReward;// hip height matches terrain
        float towardsReward;// facing consistency
        float velocityReward;// speed consistency
        float bodyStableReward;// body stability
        float footDragPenal;// no dragging
        float walklengthReward;
        float keyHipHightReward;
        float keyHipRotReward;
        float keyRotReward = 0;
        float positionReward = 0;

        // 1: Task performance (terrain, direction, speed) -----------------------------------------
        // 1.1 hip height matches terrain
        var delatHeight = Mathf.Clamp(Mathf.Abs(FootHeight(hips) - IniHipHeight), 0.15f, 2)/0.2f;
        hipHeightReward = (float)Math.Exp(-delatHeight * delatHeight); 

        // 1.2 Direction alignment (dot-product based)
        Vector3 bodyForward = (spine.forward + hips.forward + chest.forward) / 3;
        Vector3 bodyUpward = (spine.up + hips.up + chest.up) / 3;
        //towardsReward = ((Vector3.Dot(bodyForward, Vector3.right) + 1) * 0.5f)*((Vector3.Dot(bodyUpward, Vector3.up) + 1) * 0.5f);
        towardsReward = (float)Math.Exp(-bodyForward.y * bodyForward.y)* (float)Math.Exp(-bodyForward.z * bodyForward.z);
        towardsReward *= (float)Math.Exp(-bodyUpward.z * bodyUpward.z) * (float)Math.Exp(-bodyUpward.z * bodyUpward.z);

        // 1.3 Speed alignment; use avg body speed and hip speed over 100 frames
        float deltaV = (velGoal - averVelocity)/0.8f; float deltaVHip = (velGoal - averVelocityHip) / 0.8f;
        velocityReward = (float)Math.Exp(-deltaV * deltaV)* (float)Math.Exp(-deltaVHip * deltaVHip);
        //Debug.Log("deltaV: "+ deltaV + " deltaVHip: " + deltaVHip);

        //1.4 Position alignment
        float deltaP = (velGoal/framerate*step + m_JdController.bodyPartsDict[hips].startingPos.x-hips.position.x) / 0.2f;
        positionReward = (float)Math.Exp(-deltaP * deltaP);
        //Debug.Log(positionReward);
        //goalReward = hipHeightReward * (float)Math.Pow(towardsReward,3)* velocityReward;
        goalReward = 0.2f*hipHeightReward + 0.2f*towardsReward +0.2f*velocityReward+0.3f* positionReward;
        goalReward = 0.2f * hipHeightReward + 0.2f * towardsReward + 0.5f * positionReward;

        //Debug.Log("hipHeightReward: "+ hipHeightReward + " towardsReward: "+ towardsReward + " velocityReward: " + velocityReward);

        // 2: Walking quality (foot contact, no dragging, distance) -------------------------------
        // 2.1 Foot contact
        if (m_JdController.bodyPartsDict[footL].groundContact.touchingGround && !m_JdController.bodyPartsDict[footR].groundContact.touchingGround)
        { bodyStableReward = -0.005f; }
        else
        { bodyStableReward = 0;}   
        // 2.2 No dragging
        footDragPenal = FootDrag();// zero when airborne; negative when dragging
        // 2.3 Cover distance
        walklengthReward = 0.005f;

        //walkAbilityReward = bodyStableReward + footDragPenal + walklengthReward;
        //Debug.Log("bodyStableReward(未触地): " + bodyStableReward + " footDragPenal: " + footDragPenal + " walklengthReward: " + walklengthReward);


        // 3: Keyframe motion similarity ---------------------------------------------------------
        KeyPhase = keyini + (int)Math.Round((double)(keyend - keyini + 1) / KeyPhaseTotal * KeyPhaseNum, 0);// dataset frame index in use, e.g., 30
        //Debug.Log("KeyPhaseTotal frames after conversion " + KeyPhaseTotal + "  current KeyPhase frame " + KeyPhase);

        // 3.1 Hip height + foot height
        float datahip = observationLogList[KeyPhase][0]*0.01f+IniHipHeight;// demo ground clearance
        var delta1 = (FootHeight(hips) - datahip) / 0.4f;
        keyHipHightReward = (float)Math.Exp(-delta1 * delta1);
        //Debug.Log("Dataset hip height " + datahip +" current hip height " + FootHeight(hips) + "  keyHipHightReward " + keyHipHightReward);

 
        // Try using foot heights
        datahip = observationLogList[KeyPhase][1] * 0.3f + IniFootHeightLeft;// demo ground clearance
        delta1 = (FootHeight(hips) - datahip) / 0.4f;
        keyHipHightReward += (float)Math.Exp(-delta1 * delta1);
        datahip = observationLogList[KeyPhase][2] * 0.3f + IniFootHeightRight;// demo ground clearance
        delta1 = (FootHeight(hips) - datahip) / 0.4f;
        keyHipHightReward += (float)Math.Exp(-delta1 * delta1);
        //Debug.Log("Dataset hip height " + datahip + " current hip height " + FootHeight(footR) + "  keyHipHightReward " + keyHipHightReward);


        // 3.2 Hip rotation
        Quaternion datahipR = new Quaternion(observationLogList[KeyPhase][11], observationLogList[KeyPhase][12], observationLogList[KeyPhase][13], observationLogList[KeyPhase][14]);
        var DeltaMagnitude = DeltaAngle(datahipR.eulerAngles, hips.rotation.eulerAngles) / 8;
        keyHipRotReward = (float)Math.Exp(-DeltaMagnitude * DeltaMagnitude);
        //Debug.Log("Dataset rotation angle " + datahipR.eulerAngles + " current rotation angle " + hips.rotation.eulerAngles + "  keyHipRotReward " + keyHipRotReward);

        // 3.3 Other joint rotations
        var bpDict = m_JdController.bodyPartsDict;
        List<BodyPart> bpList = new List<BodyPart>() { bpDict[chest], bpDict[spine], bpDict[thighL], bpDict[shinL], bpDict[footL], bpDict[thighR], bpDict[shinR], bpDict[footR] };
        int numBp = 21; 
        foreach (var bp in bpList)
        {
            Transform trans = bp.rb.GetComponent<Transform>();

            Quaternion dataR = new Quaternion(observationLogList[KeyPhase][numBp], observationLogList[KeyPhase][numBp+1], observationLogList[KeyPhase][numBp+2], observationLogList[KeyPhase][numBp+3]);
            DeltaMagnitude = DeltaAngle(dataR.eulerAngles, trans.localRotation.eulerAngles) / 15;
            keyRotReward += (float)Math.Exp(-DeltaMagnitude * DeltaMagnitude);
            numBp += 11;
            //Debug.Log("数据集中旋转角" + dataR.eulerAngles + "当前旋转角" + trans.localRotation.eulerAngles + "  keyRotReward" + keyRotReward);

        }
        if (step == 1) { keyRotReward = 0; }
        keyFrameReward = (keyHipHightReward + keyHipRotReward + keyRotReward) /11; 

        KeyPhaseNum++;// advance current phase index
        if (KeyPhaseNum == KeyPhaseTotal)// phase complete
        { KeyPhaseNum = 0; }

        //totalReward = 0.3f*goalReward + 0.2f*walkAbilityReward + 0.5f * keyFrameReward;
        totalReward = 0.5f * goalReward +  0.5f * keyFrameReward;
        //Debug.Log("totalReward: "+ totalReward);
        AddReward(totalReward);

        statsRecorder.Add("1- goal Reward", goalReward, StatAggregationMethod.Average);
        statsRecorder.Add("1.1 hip Height Reward", hipHeightReward, StatAggregationMethod.Average);
        statsRecorder.Add("1.2 towards Reward", towardsReward, StatAggregationMethod.Average);
        statsRecorder.Add("1.3 velocity Reward", velocityReward, StatAggregationMethod.Average);

        //statsRecorder.Add("2- walk Ability Reward", walkAbilityReward, StatAggregationMethod.Average);
        statsRecorder.Add("2.1 float penalty", bodyStableReward, StatAggregationMethod.Average);
        statsRecorder.Add("2.2 foot Drag Penalty", footDragPenal, StatAggregationMethod.Average);
        statsRecorder.Add("2.3 walk length", walklengthReward, StatAggregationMethod.Average);

        statsRecorder.Add("3- key Frame Reward", keyFrameReward, StatAggregationMethod.Average);
        statsRecorder.Add("3.1 key Hip Hight Reward", keyHipHightReward, StatAggregationMethod.Average);
        statsRecorder.Add("3.2 key Hip Rot Reward", keyHipRotReward, StatAggregationMethod.Average);
        statsRecorder.Add("3.3 key Rot Reward", keyRotReward, StatAggregationMethod.Average);
    }



    /// <summary>
    /// Range 0-1; equals 1 when not dragging
    /// </summary>
    /// <returns></returns>
    private float FootDrag()
    {
        float rewardDrag = 0;

        float leftpre = m_JdController.bodyPartsDict[footL].preContact ? 1 : 0;
        float rightpre = m_JdController.bodyPartsDict[footR].preContact ? 1 : 0;
        float leftnow = m_JdController.bodyPartsDict[footL].groundContact.touchingGround ? 1 : 0;
        float rightnow = m_JdController.bodyPartsDict[footR].groundContact.touchingGround ? 1 : 0;

        float deltaLeft = Vector3.Distance(footL.position, m_JdController.bodyPartsDict[footL].PrePosition)/0.5f;
        float deltaRight = Vector3.Distance(footR.position, m_JdController.bodyPartsDict[footR].PrePosition) /0.5f;
        
        rewardDrag -= leftpre* leftnow* (1-(float)Math.Exp(-deltaLeft * deltaLeft));
        rewardDrag -= rightpre * rightnow * (1 - (float)Math.Exp(-deltaRight * deltaRight));
        rewardDrag /= 2;
        return rewardDrag;
    }


    /// <summary>
    /// Update stored linear and angular velocity per body part
    /// </summary>
    private void JointVelocityCalculate()
    {
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {// total 9 joints; all in world frame, hip not special-cased
            if (trans != head && trans != handL && trans != handR && trans != forearmL && trans != forearmR && trans != armR && trans != armL)// skip arms/hands
            {
                // linear velocity
                m_JdController.bodyPartsDict[trans].velocity = (trans.position - m_JdController.bodyPartsDict[trans].PrePosition) / Time.fixedDeltaTime;
                // angular velocity
                m_JdController.bodyPartsDict[trans].angularVelocity = AngularVelocityCalculate(m_JdController.bodyPartsDict[trans].PreRoation, trans.rotation);
            }
        }

        // Zero upper-limb motion here
        armR.localRotation = m_JdController.bodyPartsDict[armR].startingLocalRot;
        forearmR.localRotation = m_JdController.bodyPartsDict[forearmR].startingLocalRot;
        handR.localRotation = m_JdController.bodyPartsDict[handR].startingLocalRot;
        armL.localRotation = m_JdController.bodyPartsDict[armL].startingLocalRot;
        forearmL.localRotation = m_JdController.bodyPartsDict[forearmL].startingLocalRot;
        handL.localRotation = m_JdController.bodyPartsDict[handL].startingLocalRot;

    }


    /// <summary>
    /// Compute foot height above ground
    /// </summary>
    /// <param name="foot"></param>
    /// <returns></returns>
    public float FootHeight(Transform foot)
    {
        RaycastHit[] hit;// raycast hits

        float rayDistance = 0;

        hit = Physics.RaycastAll(foot.position, Vector3.down, 500f, ~(1 << 0));// with collisions

        for (int c = 0; c < hit.Length; c++)
        {
            if (hit[c].collider.gameObject.CompareTag("ground"))// collider is ground
            { rayDistance = hit[0].distance; }
        }


        return rayDistance;
    }

    /// <summary>
    /// Compute angular velocity from quaternion delta in rad/s
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
    /// Record full-body pose: current rotation and position (world space)
    /// </summary>
    public void RecordPosture()
    {
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {
            if (trans != head)
            {
                // update position (world)
                m_JdController.bodyPartsDict[trans].PrePosition = trans.position;

                // update rotation (world)
                m_JdController.bodyPartsDict[trans].PreRoation = trans.rotation;
            }
        }

        m_JdController.bodyPartsDict[footL].preContact = m_JdController.bodyPartsDict[footL].groundContact.touchingGround;
        m_JdController.bodyPartsDict[footR].preContact = m_JdController.bodyPartsDict[footR].groundContact.touchingGround;

    }


    /// <summary>
    /// Compute difference between two rotations; return magnitude of Euler delta
    /// </summary>
    /// <param name="eulerIn"></param>
    /// <returns></returns>
    public float DeltaAngle(Vector3 datasetPre, Vector3 datasetNow)
    {
        Quaternion qPre = Quaternion.Euler(datasetPre);
        Quaternion qNow = Quaternion.Euler(datasetNow);
        Quaternion qDelta = Quaternion.Inverse(qPre) * qNow;
        Vector3 angleDelta = qDelta.eulerAngles;
        if (angleDelta.x > 180) { angleDelta.x -= 360; }
        if (angleDelta.y > 180) { angleDelta.y -= 360; }
        if (angleDelta.z > 180) { angleDelta.z -= 360; }

        return angleDelta.magnitude;
    }



    public class JsonDataLog
    {
        public List<List<float>> observationLog = new List<List<float>>();
        public List<List<float>> actionLog = new List<List<float>>();
    }


}

