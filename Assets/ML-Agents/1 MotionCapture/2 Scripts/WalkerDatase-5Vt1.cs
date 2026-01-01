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

/// Train the humanoid controller: compute linear/angular velocity targets, output desired angles, and log observations/actions

public class WalkerDataset : Agent
{

    public float framerate;

    /// <summary>
    /// When true, record current observation/action. Only use during inference (with a trained network); uses full sequence length. During training, truncates to length 250.
    /// </summary>
    public bool flagLogJson;

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

    public int velocityRange; // number of frames for average speed (default 100)

    private List<float> positionlist;
    private float angleTowards;
    private JointDriveController m_JdController;
    private float[,] motionCaptureData;
    float[,] imuFootL;
    float[,] imuFootR;
    private ImuData imuData;
    private int i;
    private int length;
    private List<string> fileList;
    public int fileNum;
    private float rewardStep;
    public StatsRecorder statsRecorder;// tensorboard output
    private float errorHipPosition;// position error
    private float errorFootPosition;
    private float errorAllRotation;
    private string dirOut;

    // logs for inference runs
    private List<List<float>> observationLogList;
    private List<List<float>> actionLogList;

    private float IniFootHeightLeft;
    private float IniFootHeightRight;
    private float IniHipHeight;

    protected override void Awake()
    {
        base.Awake();
        statsRecorder = Academy.Instance.StatsRecorder;// tensorboard output
        dirOut = Path.Combine(Application.dataPath, "ObserAction");
        Debug.Log(dirOut);
        Application.targetFrameRate = 30;
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

        // Get mocap dataset file list
        fileList = GetFiles(Path.Combine(Application.streamingAssetsPath,"JsonWithP"), ".json");
        fileList = fileList.OrderBy(p => Guid.NewGuid().ToString()).ToList();// shuffle order
        fileNum = 0;

    }


    public override void OnEpisodeBegin()
    {
        // Load mocap data sequentially
        fileNum++;
        if (fileNum == fileList.Count()) { fileNum = 0; }
        imuData = JsonConvert.DeserializeObject<ImuData>(File.ReadAllText(@fileList[fileNum]));

        // To choose a specific file, disable flagLogJson and set dirOne manually:
        // string dirOne = Application.streamingAssetsPath + "/JsonWithP/walk73.json";
        // imuData = JsonConvert.DeserializeObject<ImuData>(File.ReadAllText(@dirOne));
        // Debug.Log("Single file: "+ dirOne);

        int cut = (int)Math.Ceiling(120 / framerate);

        float[,] motionCaptureDataCut = new float[imuData.walk.Length/62/cut, 62];// downsampled full sequence
        float[,] imuFootLCut = new float[imuData.walk.Length / 62 / cut, 3];
        float[,] imuFootRCut = new float[imuData.walk.Length / 62 / cut, 3];
        float[,] imuComCut = new float[imuData.walk.Length / 62 / cut, 3];

        // Downsample: keep every "cut" frame
        for (int m = 0; m < motionCaptureDataCut.Length/62; m++)
        {
            for (int n = 0; n < 62; n++)
            {
                motionCaptureDataCut[m, n] = imuData.walk[cut*m, n];
            }
            for (int y = 0; y < 3; y++)
            {
                imuFootLCut[m, y] = imuData.footL[cut * m, y];
                imuFootRCut[m, y] = imuData.footR[cut * m, y];
                imuComCut[m, y] = imuData.comPosition[cut * m, y];
            }

        }
        if (flagLogJson)
        {   motionCaptureData = motionCaptureDataCut; imuFootL = imuFootLCut; imuFootR = imuFootRCut; }
        else
        {
            int ini = UnityEngine.Random.Range(1, motionCaptureDataCut.Length / 62 - 63);// start index
            motionCaptureData = new float[62, 62];
            imuFootL = new float[62, 3]; imuFootR = new float[62, 3]; 
            for (int m = 0; m < 62; m++)
            {
                for (int n = 0; n < 62; n++)
                {
                    motionCaptureData[m, n] = motionCaptureDataCut[m + ini, n];
                }
                for (int y = 0; y < 3; y++)
                {
                    imuFootL[m, y] = imuFootLCut[m + ini, y];
                    imuFootR[m, y] = imuFootRCut[m + ini, y];
                }
            }
            for (int m = 0; m < 62; m++)// zero initial horizontal position
            {
                motionCaptureData[m, 0] -= motionCaptureDataCut[ini, 0];  motionCaptureData[m, 2] -= motionCaptureDataCut[ini, 2];
                imuFootL[m, 0] -= motionCaptureDataCut[ini, 0]; imuFootL[m, 2] -= motionCaptureDataCut[ini, 2];
                imuFootR[m, 0] -= motionCaptureDataCut[ini, 0]; imuFootR[m, 2] -= motionCaptureDataCut[ini, 2];
            }
        }

        // Initialize dataset state: frame index i, length, facing angle
        i = 0;
        length = motionCaptureData.Length / 62;
        angleTowards = Vector3.Angle(new Vector3(motionCaptureData[length - 1, 0], 0, motionCaptureData[length - 1, 2]), Vector3.right);// how much to rotate to face dataset direction
        // Initialize pose
        UpdatePosture(i);// apply frame 0 mocap data for initial pose/hip position
        RecordPosture();// store current rotation/position (world space)

        i++;

        rewardStep = 0;
        // Initialize tensorboard tracking
        errorHipPosition = 0;// position error
        errorAllRotation = 0;
        errorFootPosition = 0;

        foreach (var bp in m_JdController.bodyPartsDict.Values)
        { bp.rotationError = 0; }
        positionlist = new List<float>();

        // Create empty logs if recording
        if (flagLogJson)
        {
            observationLogList = new List<List<float>>();
            actionLogList = new List<List<float>>();
            IniFootHeightLeft = FootHeight(footL);
            IniFootHeightRight = FootHeight(footR);
            IniHipHeight = FootHeight(hips);
        }

        positionlist.Add(m_JdController.bodyPartsDict[hips].startingPos.x);
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        // Update reward before reading next clip
        if (i >= 2)
        { rewardStep = UpdateReward(); }
        else { rewardStep = 0; }

        UpdatePosture(i);// update full-body pose + hip position
        positionlist.Add(hips.position.x);

        // Rigidbody info for each body segment
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {// 9 joints used; each has 4+4+3+3+1=15 observations except hip
            if (trans != head && trans != handL && trans != handR && trans != forearmL && trans != forearmR && trans != armR && trans != armL)// skip arms/hands
            {
                // Add current and next local rotation to observations
                m_JdController.bodyPartsDict[trans].NextLocalRoation = RotationCalculate(trans, i + 1);
                sensor.AddObservation(trans.localRotation);
                sensor.AddObservation(m_JdController.bodyPartsDict[trans].NextLocalRoation);
                sensor.AddObservation(Quaternion.FromToRotation(trans.localRotation.eulerAngles, m_JdController.bodyPartsDict[trans].NextLocalRoation.eulerAngles));

                // Current linear velocity
                m_JdController.bodyPartsDict[trans].velocity = (trans.position - m_JdController.bodyPartsDict[trans].PrePosition) / (1.0f/framerate);
                sensor.AddObservation(m_JdController.bodyPartsDict[trans].velocity);

                // Current angular velocity
                m_JdController.bodyPartsDict[trans].angularVelocity = AngularVelocityCalculate(m_JdController.bodyPartsDict[trans].PreRoation, trans.rotation);
                sensor.AddObservation(m_JdController.bodyPartsDict[trans].angularVelocity);

                // Current joint torque limit (hip has none)
                if (trans != hips)
                { sensor.AddObservation(m_JdController.bodyPartsDict[trans].currentStrength / m_JdController.maxJointForceLimit); }
            }
        }
        
        // Hip position: current offset and delta to next frame (3+3)
        sensor.AddObservation(hips.position - m_JdController.bodyPartsDict[hips].startingPos);// relative to initial pose
        sensor.AddObservation(m_JdController.bodyPartsDict[hips].startingPos + Quaternion.AngleAxis(angleTowards, Vector3.up) * new Vector3(motionCaptureData[i + 1, 0], motionCaptureData[i + 1, 1], motionCaptureData[i + 1, 2]) - hips.position);
        
        // Foot-ground contact (2)
        sensor.AddObservation(m_JdController.bodyPartsDict[footL].groundContact.touchingGround);
        sensor.AddObservation(m_JdController.bodyPartsDict[footR].groundContact.touchingGround);

        if (flagLogJson)
        {
            List<float> listA = new List<float>();

            // (hip height above ground - initial) / max height, float, 1
            listA.Add((FootHeight(hips) - IniHipHeight) / 0.01f);
            Debug.Log("(hip height delta)/max: " + (FootHeight(hips) - IniHipHeight));

            //(foot height above ground - initial)/max, float, 2
            listA.Add(Math.Max(0,FootHeight(footL) - IniFootHeightLeft) / 0.3f);
            listA.Add(Math.Max(0, FootHeight(footR) - IniFootHeightRight) / 0.3f);
            Debug.Log("(foot height delta)/max: " + (Math.Max(0, FootHeight(footL) - IniFootHeightLeft) / 0.3f));

            // Target speed minus average speed over window, float, 1
            int init ;
            if (i > velocityRange) { init = i - velocityRange; } else { init = 0; }// start index for averaging
            float averVelocity = (hips.position.x - positionlist[init]) / (1.0f / framerate * (i - init));
            float velGoal = (Quaternion.AngleAxis(angleTowards, Vector3.up) * new Vector3(motionCaptureData[i, 0]-motionCaptureData[init, 0], motionCaptureData[i, 1] - motionCaptureData[init, 1], motionCaptureData[i, 2] - motionCaptureData[init, 2])/(1.0f/framerate* (i-init))).x;
            //Debug.Log("Target speed " + velGoal);
            listA.Add(velGoal - averVelocity);
            //Debug.Log("Target speed minus average " + (velGoal - averVelocity));

            // Target speed, float, 1
            listA.Add(velGoal);

            // Body forward vs target direction difference, 4
            Vector3 bodyForward = (spine.forward + chest.forward) / 2;
            Vector3 bodyForwardGoal = Vector3.right;
            Quaternion ro = Quaternion.FromToRotation(bodyForward, bodyForwardGoal);
            listA.Add(ro.x); listA.Add(ro.y); listA.Add(ro.z); listA.Add(ro.w);
            //Debug.Log("Body forward vs target diff " + ro);
            //Debug.Log("bodyForward" + bodyForward + "  bodyForwardGoal" + bodyForwardGoal);

            // Foot-ground contact as 0/1, 2
            if (m_JdController.bodyPartsDict[footL].groundContact.touchingGround == true) { listA.Add(1); } else { listA.Add(0); }
            if (m_JdController.bodyPartsDict[footR].groundContact.touchingGround == true) { listA.Add(1); } else { listA.Add(0); }
            
            // Append per-joint info
            var bpDict = m_JdController.bodyPartsDict;
            List < BodyPart >  bpList = new List<BodyPart>() { bpDict[chest],bpDict[spine],bpDict[thighL],bpDict[shinL],bpDict[footL],bpDict[thighR],bpDict[shinR],bpDict[footR]};
            
            // First add hips (4+3+3=10)

            // Global rotation quaternion, 4
            listA.Add(hips.rotation.x); listA.Add(hips.rotation.y); listA.Add(hips.rotation.z); listA.Add(hips.rotation.w);
            //Debug.Log("Hip rotation " + hips.rotation);

            // Linear velocity, 3
            listA.Add(bpDict[hips].velocity.x); listA.Add(bpDict[hips].velocity.y); listA.Add(bpDict[hips].velocity.z);
            //Debug.Log("Linear velocity " + bpDict[hips].velocity);

            // Angular velocity, 3
            listA.Add(bpDict[hips].angularVelocity.x); listA.Add(bpDict[hips].angularVelocity.y); listA.Add(bpDict[hips].angularVelocity.z);;
            //Debug.Log("Angular velocity " + bpDict[hips].angularVelocity);

            foreach (var bp in bpList)
            {   // 8 joints; each has 4+3+3+1=11

                // Local rotation quaternion, 4
                var rot = bp.rb.GetComponent<Transform>().localRotation;
                listA.Add(rot.x); listA.Add(rot.y); listA.Add(rot.z); listA.Add(rot.w);
                //Debug.Log("Local rotation quaternion " + rot);

                // Linear velocity, 3
                listA.Add(bp.velocity.x); listA.Add(bp.velocity.y); listA.Add(bp.velocity.z);
                //Debug.Log("Linear velocity " + bp.velocity);

                // Angular velocity, 3
                listA.Add(bp.angularVelocity.x); listA.Add(bp.angularVelocity.y); listA.Add(bp.angularVelocity.z);
                //Debug.Log("Angular velocity " + bp.angularVelocity);

                // Torque limit, 1
                listA.Add(bp.currentStrength / m_JdController.maxJointForceLimit);
                //Debug.Log("Torque limit " + bp.currentStrength / m_JdController.maxJointForceLimit);
            }
            
            observationLogList.Add(listA);
        }

        RecordPosture();// refresh rotation/position (world) for next step
        RigidControl();// set rigidbody velocities (world space)

    }



    public override void OnActionReceived(ActionBuffers actions)
    {
        var bpDict = m_JdController.bodyPartsDict;
        var n = -1;

        // Map action axes to joints
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


        if (flagLogJson)
        {
            List<float> listB = new List<float>();
            for (int m = 0; m < continuousActions.Length; m++)
            { listB.Add(continuousActions[m]);}
            actionLogList.Add(listB);
        }


        if (i == length - 2)
        {
            //Debug.Log("Sequence end: "+i + "  cumulative reward: " + GetCumulativeReward());

            foreach (var bp in bpDict.Values)
            { bp.rotationError /= i - 2; }// average error

            // For inference debugging
            //Debug.Log("Rotation error: chest" + bpDict[chest].rotationError + " spine" + bpDict[spine].rotationError+" hip"+ bpDict[hips].rotationError);
            //Debug.Log("thighL" + bpDict[thighL].rotationError + " shinL" + bpDict[shinL].rotationError + " footL" + bpDict[footL].rotationError);
            //Debug.Log("thighR" + bpDict[thighR].rotationError + " shinR" + bpDict[shinR].rotationError + " footL" + bpDict[footR].rotationError);

            // Tensorboard stats
            statsRecorder.Add("Average Hip Position error", errorHipPosition / (i - 2), StatAggregationMethod.Average);
            statsRecorder.Add("Average all Roatation error", errorAllRotation / (i - 2), StatAggregationMethod.Average);
            statsRecorder.Add("Average Foot Position error", errorFootPosition / (i - 2)/2, StatAggregationMethod.Average);

            statsRecorder.Add("Average hips Roatation error", bpDict[hips].rotationError, StatAggregationMethod.Average);
            statsRecorder.Add("Average spine Roatation error", bpDict[spine].rotationError, StatAggregationMethod.Average);
            statsRecorder.Add("Average chest Roatation error", bpDict[chest].rotationError, StatAggregationMethod.Average);
            statsRecorder.Add("Average shinL Roatation error", bpDict[shinL].rotationError, StatAggregationMethod.Average);
            statsRecorder.Add("Average shinR Roatation error", bpDict[shinR].rotationError, StatAggregationMethod.Average);
            statsRecorder.Add("Average thighL Roatation error", bpDict[thighL].rotationError, StatAggregationMethod.Average);
            statsRecorder.Add("Average thighR Roatation error", bpDict[thighR].rotationError, StatAggregationMethod.Average);
            statsRecorder.Add("Average footL Roatation error", bpDict[footL].rotationError, StatAggregationMethod.Average);
            statsRecorder.Add("Average footR Roatation error", bpDict[footR].rotationError, StatAggregationMethod.Average);

            //if (flagLogJson){ SaveObsActJson(observationLogList, actionLogList, dirOut, fileList[fileNum]); }

            AddReward(rewardStep);
            EndEpisode();// end episode when sequence completes
        }
        i++;

        AddReward(rewardStep);

    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Debug-only; use inference mode for real logging
        var continuousActions = actionsOut.ContinuousActions;
        for (int n = 0; n < continuousActions.Length; n++)
        {
            continuousActions[n] = 0;
        }
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
    /// At episode end, save observation/action to json (append to 2D lists)
    /// </summary>
    /// <param name="observationLogList">observation 2D list; first dim is step index, second is data</param>
    /// <param name="actionLogList">action 2D list; first dim aligns with observation</param>
    /// <param name="dirOut">output json path + filename</param>
    /// <param name="dataFileName">dataset json file for this episode</param>
    public void SaveObsActJson(List<List<float>> observation, List<List<float>> action, string dirOut, string dataFileName)
    {
        dataFileName = dataFileName.Remove(0, 67); ;// keep only trailing filename like "walk5.json"
        JsonDataLog jsonDataLog = new JsonDataLog();// instantiate and initialize lists
        
        // Append this episode's data
        jsonDataLog.observationLog = observation;
        jsonDataLog.actionLog = action; 
      
        // Save json
        File.WriteAllText(dirOut + dataFileName, JsonConvert.SerializeObject(jsonDataLog), System.Text.Encoding.UTF8);
    }



    /// <summary>
    /// Based on mocap data, compute root-space quaternion (XYZ order)
    /// </summary>
    /// <param name="angleX"></param>
    /// <param name="angleY"></param>
    /// <param name="angleZ"></param>
    /// <returns>rotation quaternion</returns>
    private Quaternion AngleRoatation(float angleX, float angleY, float angleZ)
    {
        Quaternion RotationQ = Quaternion.Euler(new Vector3(angleX, 0, 0)) * Quaternion.Euler(new Vector3(0, angleY, 0)) * Quaternion.Euler(new Vector3(0, 0, angleZ));

        return RotationQ;
    }

    /// <summary>
    /// Apply target linear/angular velocity to rigidbodies
    /// </summary>
    public void RigidControl()
    {
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {
            if (trans != head && trans != handL && trans != handR && trans != forearmL && trans != forearmR && trans != armR && trans != armL)
            {
                // Feed values to rigid controller
                m_JdController.bodyPartsDict[trans].rb.angularVelocity = m_JdController.bodyPartsDict[trans].angularVelocity;
                m_JdController.bodyPartsDict[trans].rb.velocity = m_JdController.bodyPartsDict[trans].velocity;

            }
        }
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
        Vector3 AngularVelocityThis = angularDisplacement / (1.0f/framerate);

        return AngularVelocityThis;
    }


    /// <summary>
    /// From dataset frame and transform, compute localRotation
    /// </summary>
    /// <param name=""></param>
    public Quaternion RotationCalculate(Transform trans, int frameNumber)
    {
        Quaternion localRotation = new Quaternion();
        // Adjust forward direction if needed
        //float angleTowards = Vector3.Angle(new Vector3(motionCaptureData[length - 1, 0], 0, motionCaptureData[length - 1, 2]), Vector3.right);

        if (trans == hips) localRotation = m_JdController.bodyPartsDict[hips].startingRot * AngleRoatation(motionCaptureData[frameNumber, 3], -motionCaptureData[frameNumber, 4], -motionCaptureData[frameNumber, 5]);
        // spine covers lowerback and upperback
        else if (trans == spine) localRotation = AngleRoatation(-10, 0, 0) * AngleRoatation(motionCaptureData[frameNumber, 6], -motionCaptureData[frameNumber, 7], -motionCaptureData[frameNumber, 8]) * AngleRoatation(motionCaptureData[frameNumber, 9], -motionCaptureData[frameNumber, 10], -motionCaptureData[frameNumber, 11]);
        else if (trans == chest) localRotation = AngleRoatation(motionCaptureData[frameNumber, 12], -motionCaptureData[frameNumber, 13], -motionCaptureData[frameNumber, 14]);

        else if (trans == armR) localRotation = m_JdController.bodyPartsDict[armR].startingLocalRot;// m_JdController.bodyPartsDict[armR].startingLocalRot * AngleRoatation(-30, 0, 0) * AngleRoatation(0, 0, -90) * AngleRoatation(motionCaptureData[frameNumber, 26], motionCaptureData[frameNumber, 27], motionCaptureData[frameNumber, 28]) * AngleRoatation(0, 0, 90) * AngleRoatation(30, 0, 0);// * AngleRoatation(10, 10, 0); // x rotates hand inward; y swings arm back; z toward torso
        else if (trans == forearmR) localRotation = m_JdController.bodyPartsDict[forearmR].startingLocalRot;//AngleRoatation(0, -motionCaptureData[frameNumber, 29], 0); // for rradius
        else if (trans == handR) localRotation = m_JdController.bodyPartsDict[handR].startingLocalRot;//AngleRoatation(motionCaptureData[frameNumber, 30], 0, 0) * AngleRoatation(0, 0, -motionCaptureData[frameNumber, 31]) * AngleRoatation(0, -motionCaptureData[frameNumber, 32], 0); // for rwrist+rhand
        else if (trans == armL) localRotation = m_JdController.bodyPartsDict[armL].startingLocalRot;//m_JdController.bodyPartsDict[armL].startingLocalRot * AngleRoatation(-30, 0, 0) * AngleRoatation(0, 0, 90) * AngleRoatation(motionCaptureData[frameNumber, 38], motionCaptureData[frameNumber, 39], motionCaptureData[frameNumber, 40]) * AngleRoatation(0, 0, -90) * AngleRoatation(30, 0, 0);// * AngleRoatation(20, 0, 10);
        else if (trans == forearmL) localRotation = m_JdController.bodyPartsDict[forearmL].startingLocalRot;//AngleRoatation(0, motionCaptureData[frameNumber, 41], 0);
        else if (trans == handL) localRotation = m_JdController.bodyPartsDict[handL].startingLocalRot;//AngleRoatation(-motionCaptureData[frameNumber, 42], 0, 0) * AngleRoatation(0, 0, motionCaptureData[frameNumber, 43]) * AngleRoatation(0, -motionCaptureData[frameNumber, 44], 0);

        else if (trans == thighR) localRotation = AngleRoatation(0, 0, 20) * AngleRoatation(motionCaptureData[frameNumber, 48], -motionCaptureData[frameNumber, 49], -motionCaptureData[frameNumber, 50]);
        else if (trans == shinR) localRotation = AngleRoatation(motionCaptureData[frameNumber, 51], 0, 0);
        else if (trans == footR) localRotation = m_JdController.bodyPartsDict[footR].startingLocalRot * AngleRoatation(motionCaptureData[frameNumber, 52], 0, -motionCaptureData[frameNumber, 53]);
        else if (trans == thighL) localRotation = AngleRoatation(0, 0, -20) * AngleRoatation(motionCaptureData[frameNumber, 55], -motionCaptureData[frameNumber, 56], -motionCaptureData[frameNumber, 57]);
        else if (trans == shinL) localRotation = AngleRoatation(motionCaptureData[frameNumber, 58], 0, 0);
        else if (trans == footL) localRotation = m_JdController.bodyPartsDict[footL].startingLocalRot * AngleRoatation(motionCaptureData[frameNumber, 59], 0, -motionCaptureData[frameNumber, 60]);
        return localRotation;

    }

    /// <summary>
    /// Update full-body pose using frame "frame" mocap data
    /// </summary>
    /// <param name="frame"></param>
    public void UpdatePosture(int frame)
    {
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {
            if (trans != head && trans != hips)// others use local rotation
            {
                trans.localRotation = RotationCalculate(trans, frame);
            }
            else if (trans == hips)// hip uses global rotation
            {
                trans.rotation = RotationCalculate(trans, frame);
            }
        }
        hips.position = m_JdController.bodyPartsDict[hips].startingPos + Quaternion.AngleAxis(angleTowards, Vector3.up) * new Vector3(motionCaptureData[frame, 0], motionCaptureData[frame, 1], motionCaptureData[frame, 2]);
        head.rotation = m_JdController.bodyPartsDict[head].startingRot;
        footL.position = m_JdController.bodyPartsDict[footL].startingPos + new Vector3(imuFootL[i, 0], imuFootL[i, 1], imuFootL[i, 2]);
        footR.position = m_JdController.bodyPartsDict[footR].startingPos + new Vector3(imuFootR[i, 0], imuFootR[i, 1], imuFootR[i, 2]);
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
    }

    /// <summary>
    /// Compare current values to mocap frame i
    /// </summary>
    public float UpdateReward()
    {
        float hipPositionReward;
        float bodyRotationReward = 0;// local rotation of each body segment
        float walkLengthReward = 0;
        float footPositionReward = 0;
        float rewardTotal;
        float errorTotal = 0;

        // Common intermediates
        Vector3 datasetPre;// previous frame (dataset)
        Vector3 datasetNow;// current frame (dataset)
        Vector3 datasetSim;// simulated value
        float maxMagnitude;// max allowed delta
        float DeltaMagnitude;// actual delta

        //hipPositionReward--------------------------------------------------
        maxMagnitude = 0.01f;
        //datasetPre = Quaternion.AngleAxis(angleTowards, Vector3.up) * new Vector3(motionCaptureData[i-1, 0], motionCaptureData[i-1, 1], motionCaptureData[i-1, 2]);
        datasetNow = Quaternion.AngleAxis(angleTowards, Vector3.up) * new Vector3(motionCaptureData[i, 0], motionCaptureData[i, 1], motionCaptureData[i, 2]);
        datasetSim = hips.position - m_JdController.bodyPartsDict[hips].startingPos;

        DeltaMagnitude = Vector3.Distance(datasetSim, datasetNow)/maxMagnitude;
        hipPositionReward = (float)Math.Exp(-DeltaMagnitude * DeltaMagnitude);

        errorHipPosition += Vector3.Distance(datasetSim, datasetNow);// accumulate error

        //footPositionReward-----------------------------------------------------
        maxMagnitude = 0.03f;
        //datasetPre = new Vector3(imuFootL[i - 1, 0], imuFootL[i - 1, 1], imuFootL[i - 1, 2]);
        datasetNow = new Vector3(imuFootL[i, 0], imuFootL[i, 1], imuFootL[i, 2]);
        datasetSim = footL.position - m_JdController.bodyPartsDict[footL].startingPos;
        DeltaMagnitude = Vector3.Distance(datasetSim, datasetNow) / maxMagnitude;
        footPositionReward += (float)Math.Exp(-DeltaMagnitude * DeltaMagnitude);
        errorFootPosition += Vector3.Distance(datasetSim, datasetNow);

        //datasetPre = new Vector3(imuFootR[i - 1, 0], imuFootR[i - 1, 1], imuFootR[i - 1, 2]);
        datasetNow = new Vector3(imuFootR[i, 0], imuFootR[i, 1], imuFootR[i, 2]);
        datasetSim = footR.position - m_JdController.bodyPartsDict[footR].startingPos;
        DeltaMagnitude = Vector3.Distance(datasetSim, datasetNow) / maxMagnitude;
        footPositionReward += (float)Math.Exp(-DeltaMagnitude * DeltaMagnitude);

        footPositionReward /= 2;

        errorFootPosition += Vector3.Distance(datasetSim, datasetNow);

        //bodyRotationReward-------------------------------------------------------
        int m = 0;// joint counter
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {
            if (trans != head && trans != handL && trans != handR && trans!= forearmL && trans != forearmR && trans != armR && trans != armR)
            {

                if (trans == hips)
                { datasetSim = trans.rotation.eulerAngles; }
                else
                { datasetSim = trans.localRotation.eulerAngles; }

                datasetNow = RotationCalculate(trans, i).eulerAngles;
                datasetPre = RotationCalculate(trans, i-1).eulerAngles;
                maxMagnitude = 8;

                // Exponential form per reference
                DeltaMagnitude = DeltaAngle(datasetSim, datasetNow) / maxMagnitude;
                bodyRotationReward += (float)Math.Exp(-DeltaMagnitude * DeltaMagnitude);

                errorTotal += DeltaAngle(datasetSim, datasetNow) ;
                //Debug.Log("DeltaAngle(datasetPre, datasetNow) " + DeltaAngle(datasetPre, datasetNow));
                //Debug.Log("DeltaAngle(datasetSim, datasetNow) " + DeltaAngle(datasetSim, datasetNow));
                m_JdController.bodyPartsDict[trans].rotationError += DeltaAngle(datasetSim, datasetNow);

                m++;
            }
        }
        bodyRotationReward /= m;

        // For tensorboard logging
        errorTotal /= m; 
        errorAllRotation += errorTotal;

        //walkLengthReward: small penalty if both feet airborne
        if (!m_JdController.bodyPartsDict[footL].groundContact.touchingGround && !m_JdController.bodyPartsDict[footR].groundContact.touchingGround)
        {
            walkLengthReward = - 0.01f;
            //Debug.Log("Both feet airborne: " + fileList[fileNum]);
        }

        // Overall reward (could be weighted or multiplied)
        //rewardTotal = 0.1f*hipPositionReward + 0.1f * footPositionReward  + 0.8f * bodyRotationReward + walkLengthReward;
        rewardTotal = hipPositionReward * footPositionReward * bodyRotationReward + walkLengthReward;
        return rewardTotal;
   
}


    /// <summary>
    /// Compute rotation difference magnitude in Euler space
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


    /// <summary>
    /// Parsed json data
    /// </summary>
    public class ImuData
    {
        public float[,] walk;
        public float[,] footL;
        public float[,] footR;
        public float[,] comPosition;
    }


    public class JsonDataLog
    {
        public List<List<float>> observationLog = new List<List<float>>();
        public List<List<float>> actionLog = new List<List<float>>();
    }

    /// <summary>
    /// Get files matching suffix (non-recursive)
    /// </summary>
    /// <param name="path"></param>
    /// <param name="suffix"></param>
    /// <returns></returns>
    private List<string> GetFiles(string path, string suffix)
    {
        string filename;
        List<string> fileList = new List<string>();
        DirectoryInfo dir = new DirectoryInfo(path);
        FileInfo[] file = dir.GetFiles().OrderByDescending(n => n.LastWriteTime).ToArray();

        foreach (FileInfo f in file)
        {
            filename = f.FullName;
            if (filename.EndsWith(suffix))// check suffix and add full path
            {
                fileList.Add(filename);
            }
        }
        return fileList;
    }

}
