using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using BodyPart = Unity.MLAgentsExamples.BodyPart;
using Random = UnityEngine.Random;
using Newtonsoft.Json;
using System.IO;


public class WalkerAgent: Agent
{
    [Header("Walk Speed")]
    [Range(0.3f, 2)]
    [SerializeField]
    //The walking speed to try and achieve
    private float m_TargetWalkingSpeed = 2;

    public float MTargetWalkingSpeed // property
    {
        get { return m_TargetWalkingSpeed; }
        set { m_TargetWalkingSpeed = Mathf.Clamp(value, .2f, m_maxWalkingSpeed); }
    }

    const float m_maxWalkingSpeed = 2; //The max walking speed

    //Should the agent sample a new goal velocity each episode?
    //If true, walkSpeed will be randomly set between zero and m_maxWalkingSpeed in OnEpisodeBegin()
    //If false, the goal velocity will be walkingSpeed
    public bool randomizeWalkSpeedEachEpisode;

    //The direction an agent will walk during training.
    private Vector3 m_WorldDirToWalk = Vector3.right;

    [Header("Target To Walk Towards")] public Transform target; //Target the agent will walk towards during training.

    [Header("Body Parts")] public Transform hips;
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


    //This will be used as a stabilized model space reference point for observations
    //Because ragdolls can move erratically during training, using a stabilized reference transform improves learning
    OrientationCubeController m_OrientationCube;

    //The indicator graphic gameobject that points towards the target
    JointDriveController m_JdController;
    EnvironmentParameters m_ResetParams;

    private float IniFootHeightLeft;
    private float IniFootHeightRight;
    private float IniHipHeight;
    private float averVelocityHip;

    //记录训练后inference内容的列表
    private List<List<float>> observationLogList;

    public StatsRecorder statsRecorder;//tensorboard输出
    private string dirOut;
    private JsonDataLog jsonDataLog;

    private int step;//记录位于episode的第几步
    public int velocityRange; //计算平均速度时取多少个点，默认100
    private List<float> positionlist;
    float averVelocity;

    float dataV;//demo数据中的最后一帧的目标速度
    int keyini; int keyend;
    int KeyPhaseNum;//当前进行到自己的第几个，如1-27
    int KeyPhaseTotal;//一共用几帧，如27，数据集中是37
    int KeyPhase;//当前用数据集中第几帧，如30

    float framerate = 150;

    protected override void Awake()
    {
        base.Awake();
        statsRecorder = Academy.Instance.StatsRecorder;//tensorboard输出
        //一次只录制一组动作。如果需要更多，在这个脚本里改，加一个计数
        dirOut = Path.Combine(Application.dataPath, "ObserAction/walk73.json");
        Time.fixedDeltaTime = 1.0f / framerate;//物理帧时长
        keyini = 6; keyend = 40;

    }

    public override void Initialize()
    {
        m_OrientationCube = GetComponentInChildren<OrientationCubeController>();

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

        m_ResetParams = Academy.Instance.EnvironmentParameters;

        SetResetParameters();

        //读取json文件
        jsonDataLog = JsonConvert.DeserializeObject<JsonDataLog>(File.ReadAllText(@dirOut));
        observationLogList = jsonDataLog.observationLog;//这些都是按30hz保存的

        dataV = observationLogList[observationLogList.Count - 1][4];//目标速度（对本demo是固定的）

    }

    /// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        //Reset all of the body parts
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }


        m_OrientationCube.UpdateOrientation(hips, target);

        //Set our goal walking speed
        MTargetWalkingSpeed =
            randomizeWalkSpeedEachEpisode ? Random.Range(0.2f, m_maxWalkingSpeed) : MTargetWalkingSpeed;

        SetResetParameters();

        IniFootHeightLeft = FootHeight(footL);
        IniFootHeightRight = FootHeight(footR);
        IniHipHeight = FootHeight(hips);

        step = 1;
        KeyPhaseNum = 0;
        KeyPhaseTotal = (int)Math.Round(dataV / MTargetWalkingSpeed * (keyend - keyini + 1), 0);//数据集中是37对应于这个速度的几帧


    }

    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        //GROUND CHECK
        sensor.AddObservation(bp.groundContact.touchingGround); // Is this bp touching the ground

        //Get velocities in the context of our orientation cube's space
        //Note: You can get these velocities in world space as well but it may not train as well.
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.velocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity));

        //Get position relative to hips in the context of our orientation cube's space
        sensor.AddObservation(bp.rb.position - hips.position);

        if (bp.rb.transform != hips && bp.rb.transform != handL && bp.rb.transform != handR)
        {
            sensor.AddObservation(bp.rb.transform.localRotation);
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        var cubeForward = m_WorldDirToWalk;

        //velocity we want to match
        var velGoal = cubeForward * MTargetWalkingSpeed;
        //ragdoll's avg vel
        var avgVel = GetAvgVelocity();

        //current ragdoll velocity. normalized
        sensor.AddObservation(Vector3.Distance(velGoal, avgVel));
        //avg body vel relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));
        //vel goal relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal));

        //rotation deltas
        sensor.AddObservation(Quaternion.FromToRotation(hips.forward, cubeForward));
        sensor.AddObservation(Quaternion.FromToRotation(head.forward, cubeForward));

        //Position of target position relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformPoint(target.transform.position));

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }
        //Debug.Log("Observation:" + shinR.localRotation.eulerAngles.ToString());
        
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        //strength_test = m_JdController.bodyPartsDict[shinR].currentStrength;
        //Debug.Log("observation:" + strength_test.ToString());
        m_JdController.bodyPartsDict[footL].preContact = m_JdController.bodyPartsDict[footL].groundContact.touchingGround;
        m_JdController.bodyPartsDict[footR].preContact = m_JdController.bodyPartsDict[footR].groundContact.touchingGround;

        var bpDict = m_JdController.bodyPartsDict;
        var i = -1;

        var continuousActions = actionBuffers.ContinuousActions;
        bpDict[chest].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[spine].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);

        bpDict[thighL].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[thighR].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[shinL].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[shinR].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[footR].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[footL].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);

        bpDict[armL].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[armR].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[forearmL].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[forearmR].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[head].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);

        //update joint strength settings
        bpDict[chest].SetJointStrength(continuousActions[++i]);
        bpDict[spine].SetJointStrength(continuousActions[++i]);
        bpDict[head].SetJointStrength(continuousActions[++i]);
        bpDict[thighL].SetJointStrength(continuousActions[++i]);
        bpDict[shinL].SetJointStrength(continuousActions[++i]);
        bpDict[footL].SetJointStrength(continuousActions[++i]);
        bpDict[thighR].SetJointStrength(continuousActions[++i]);
        bpDict[shinR].SetJointStrength(continuousActions[++i]);
        bpDict[footR].SetJointStrength(continuousActions[++i]);
        bpDict[armL].SetJointStrength(continuousActions[++i]);
        bpDict[forearmL].SetJointStrength(continuousActions[++i]);
        bpDict[armR].SetJointStrength(continuousActions[++i]);
        bpDict[forearmR].SetJointStrength(continuousActions[++i]);

        //Debug.Log("TargetRotation:" + bpDict[shinR].currentEularJointRotation.ToString());
    }


    void FixedUpdate()
    {
        m_OrientationCube.UpdateOrientation(hips, target);

        // 碰到目标结束训练
        if ((target.position - hips.position).magnitude < 2)
        { EndEpisode(); }

        var cubeForward = m_WorldDirToWalk;

        // Set reward for this step according to mixture of the following elements.
        // a. Match target speed
        //This reward will approach 1 if it matches perfectly and approach zero as it deviates
        var matchSpeedReward = GetMatchingVelocityReward(cubeForward * MTargetWalkingSpeed, GetAvgVelocity());

        // b. Rotation alignment with target direction.
        //This reward will approach 1 if it faces the target direction perfectly and approach zero as it deviates
        var lookAtTargetReward = (Vector3.Dot(cubeForward, head.forward) + 1) * .5F;


        float lateralMovement = (hips.position.z - m_JdController.bodyPartsDict[hips].startingPos.z) / 2;
        float keyHipHightReward = (float)Math.Exp(-lateralMovement * lateralMovement);


        float rewardKey = TotalRewardCalculate();

        AddReward(matchSpeedReward * lookAtTargetReward * keyHipHightReward * 0.5f+0.5f* rewardKey);

        statsRecorder.Add("3.3 match Speed Reward", matchSpeedReward, StatAggregationMethod.Average);
        statsRecorder.Add("3.3 look At Target Reward", lookAtTargetReward, StatAggregationMethod.Average);

        //Debug.Log("matchSpeedReward * lookAtTargetReward * keyHipHightReward" + matchSpeedReward * lookAtTargetReward * keyHipHightReward);

        

        armL.rotation = m_JdController.bodyPartsDict[armL].startingRot;
        armR.rotation = m_JdController.bodyPartsDict[armR].startingRot;
        forearmL.rotation = m_JdController.bodyPartsDict[forearmL].startingRot;
        forearmR.rotation = m_JdController.bodyPartsDict[forearmR].startingRot;
        handL.rotation = m_JdController.bodyPartsDict[handL].startingRot;
        handR.rotation = m_JdController.bodyPartsDict[handR].startingRot;

        step++;

        KeyPhaseNum++;//当前进行到自己的第几个，如1-27，每次加一
        if (KeyPhaseNum == KeyPhaseTotal)//当前phase结束
        { KeyPhaseNum = 0; }
        Debug.Log(matchSpeedReward);

    }

    public float TotalRewardCalculate()
    {
        float totalReward;
        float goalReward;
        float walkAbilityReward;
        float keyFrameReward;

        float hipHeightReward;//hip高度与地形高度一致
        float footDragPenal;//不拖地
        float walklengthReward;
        float keyHipHightReward;
        float keyHipRotReward;
        float keyRotReward = 0;
        float positionReward = 0;

        // 1: 任务完成水平（包括走地形、方向、速度）-------------------------------------------------------------------------
        // 1.1 hip高度与地形高度一致
        var delatHeight = Mathf.Clamp(Mathf.Abs(FootHeight(hips) - IniHipHeight), 0.15f, 2) / 0.2f;
        hipHeightReward = (float)Math.Exp(-delatHeight * delatHeight);
        //1.4 位置一致
        float deltaP = (MTargetWalkingSpeed / framerate * step + m_JdController.bodyPartsDict[hips].startingPos.x - hips.position.x) / 0.2f;
        positionReward = (float)Math.Exp(-deltaP * deltaP);
        goalReward = 0.5f * hipHeightReward + 0.5f * positionReward;


        // 2: 行走水平（双脚触地、脚步不拖行、走得远）------------------------------------------------------------------------
        // 2.2 不拖地
        footDragPenal = FootDrag();//不着地时为零，拖动时为负值
        // 2.3 走得远
        walklengthReward = 0.005f;
        walkAbilityReward = footDragPenal + walklengthReward;


        // 3：关键帧动作相似度------------------------------------------------------------------------
        KeyPhase = keyini + (int)Math.Round((double)(keyend - keyini + 1) / KeyPhaseTotal * KeyPhaseNum, 0);//当前用数据集中第几帧，如30

        // 3.1 hip高度+足部高度
        float datahip = observationLogList[KeyPhase][0] * 0.01f + IniHipHeight;//demo中的离地高度
        var delta1 = (FootHeight(hips) - datahip) / 0.4f;
        keyHipHightReward = (float)Math.Exp(-delta1 * delta1);
        //Debug.Log("数据集中离地高度" + datahip +"当前离地高度"+ FootHeight(hips) + "  keyHipHightReward" + keyHipHightReward);


        //换为足部高度试一下
        datahip = observationLogList[KeyPhase][1] * 0.3f + IniFootHeightLeft;//demo中的离地高度
        delta1 = (FootHeight(hips) - datahip) / 0.4f;
        keyHipHightReward += (float)Math.Exp(-delta1 * delta1);
        datahip = observationLogList[KeyPhase][2] * 0.3f + IniFootHeightRight;//demo中的离地高度
        delta1 = (FootHeight(hips) - datahip) / 0.4f;
        keyHipHightReward += (float)Math.Exp(-delta1 * delta1);
        //Debug.Log("数据集中离地高度" + datahip + "当前离地高度"+ FootHeight(footR) + "  keyHipHightReward" + keyHipHightReward);


        // 3.2 hip旋转角
        Quaternion datahipR = new Quaternion(observationLogList[KeyPhase][11], observationLogList[KeyPhase][12], observationLogList[KeyPhase][13], observationLogList[KeyPhase][14]);
        var DeltaMagnitude = DeltaAngle(datahipR.eulerAngles, hips.rotation.eulerAngles) / 8;
        keyHipRotReward = (float)Math.Exp(-DeltaMagnitude * DeltaMagnitude);
        //Debug.Log("数据集中旋转角" + datahipR.eulerAngles + "当前旋转角"+ hips.rotation.eulerAngles + "  keyHipRotReward" + keyHipRotReward);

        // 3.3 其余关节旋转角
        var bpDict = m_JdController.bodyPartsDict;
        List<BodyPart> bpList = new List<BodyPart>() { bpDict[chest], bpDict[spine], bpDict[thighL], bpDict[shinL], bpDict[footL], bpDict[thighR], bpDict[shinR], bpDict[footR] };
        int numBp = 21;
        foreach (var bp in bpList)
        {
            Transform trans = bp.rb.GetComponent<Transform>();

            Quaternion dataR = new Quaternion(observationLogList[KeyPhase][numBp], observationLogList[KeyPhase][numBp + 1], observationLogList[KeyPhase][numBp + 2], observationLogList[KeyPhase][numBp + 3]);
            DeltaMagnitude = DeltaAngle(dataR.eulerAngles, trans.localRotation.eulerAngles) / 15;
            keyRotReward += (float)Math.Exp(-DeltaMagnitude * DeltaMagnitude);
            numBp += 11;
            //Debug.Log("数据集中旋转角" + dataR.eulerAngles + "当前旋转角" + trans.localRotation.eulerAngles + "  keyRotReward" + keyRotReward);

        }
        if (step == 1) { keyRotReward = 0; }
        keyFrameReward = (keyHipHightReward + keyHipRotReward + keyRotReward) / 11;



        totalReward = 0.2f*goalReward + 0.1f*walkAbilityReward + 0.7f * keyFrameReward;

        statsRecorder.Add("1- goal Reward", goalReward, StatAggregationMethod.Average);
        statsRecorder.Add("1.1 hip Height Reward", hipHeightReward, StatAggregationMethod.Average);
        statsRecorder.Add("1.4 position Reward", positionReward, StatAggregationMethod.Average);

        statsRecorder.Add("2- walk Ability Reward", walkAbilityReward, StatAggregationMethod.Average);
        statsRecorder.Add("2.2 foot Drag Penalty", footDragPenal, StatAggregationMethod.Average);
        statsRecorder.Add("2.3 walk length", walklengthReward, StatAggregationMethod.Average);

        statsRecorder.Add("3- key Frame Reward", keyFrameReward, StatAggregationMethod.Average);
        statsRecorder.Add("3.1 key Hip Hight Reward", keyHipHightReward, StatAggregationMethod.Average);
        statsRecorder.Add("3.2 key Hip Rot Reward", keyHipRotReward, StatAggregationMethod.Average);
        statsRecorder.Add("3.3 key Rot Reward", keyRotReward, StatAggregationMethod.Average);

        return totalReward;

    }

    public float FootHeight(Transform foot)
    {
        RaycastHit[] hit;//检测射线碰撞

        float rayDistance = 0;

        hit = Physics.RaycastAll(foot.position, Vector3.down, 500f, ~(1 << 0));//有碰撞物体

        for (int c = 0; c < hit.Length; c++)
        {
            if (hit[c].collider.gameObject.CompareTag("ground"))//碰撞物体为地面
            { rayDistance = hit[0].distance; }
        }


        return rayDistance;
    }

    /// <summary>
    /// 计算两个旋转间差距，返回欧拉角距离的模长
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



    //Returns the average velocity of all of the body parts
    //Using the velocity of the hips only has shown to result in more erratic movement from the limbs, so...
    //...using the average helps prevent this erratic movement
    Vector3 GetAvgVelocity()
    {
        Vector3 velSum = Vector3.zero;

        //ALL RBS
        int numOfRb = 0;
        foreach (var item in m_JdController.bodyPartsList)
        {
            numOfRb++;
            velSum += item.rb.velocity;
        }

        var avgVel = velSum / numOfRb;
        return avgVel;
    }

    //normalized value of the difference in avg speed vs goal walking speed.
    public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
    {
        //distance between our actual velocity and goal velocity
        var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, MTargetWalkingSpeed);

        //return the value on a declining sigmoid shaped curve that decays from 1 to 0
        //This reward will approach 1 if it matches perfectly and approach zero as it deviates
        return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / MTargetWalkingSpeed, 2), 2);
    }


    public void SetTorsoMass()
    {
        m_JdController.bodyPartsDict[chest].rb.mass = m_ResetParams.GetWithDefault("chest_mass", 8);
        m_JdController.bodyPartsDict[spine].rb.mass = m_ResetParams.GetWithDefault("spine_mass", 8);
        m_JdController.bodyPartsDict[hips].rb.mass = m_ResetParams.GetWithDefault("hip_mass", 8);
    }

    public void SetResetParameters()
    {
        SetTorsoMass();
        m_JdController.bodyPartsDict[footL].preContact = m_JdController.bodyPartsDict[footL].groundContact.touchingGround;
        m_JdController.bodyPartsDict[footR].preContact = m_JdController.bodyPartsDict[footR].groundContact.touchingGround;
    }


    /// <summary>
    /// 范围0-1，不拖地时为1
    /// </summary>
    /// <returns></returns>
    private float FootDrag()
    {
        float rewardDrag = 0;

        float leftpre = m_JdController.bodyPartsDict[footL].preContact ? 1 : 0;
        float rightpre = m_JdController.bodyPartsDict[footR].preContact ? 1 : 0;
        float leftnow = m_JdController.bodyPartsDict[footL].groundContact.touchingGround ? 1 : 0;
        float rightnow = m_JdController.bodyPartsDict[footR].groundContact.touchingGround ? 1 : 0;

        float deltaLeft = Vector3.Distance(footL.position, m_JdController.bodyPartsDict[footL].PrePosition) / 0.5f;
        float deltaRight = Vector3.Distance(footR.position, m_JdController.bodyPartsDict[footR].PrePosition) / 0.5f;

        rewardDrag -= leftpre * leftnow * (1 - (float)Math.Exp(-deltaLeft * deltaLeft));
        rewardDrag -= rightpre * rightnow * (1 - (float)Math.Exp(-deltaRight * deltaRight));
        rewardDrag /= 2;
        return rewardDrag;
    }

    public class JsonDataLog
    {
        public List<List<float>> observationLog = new List<List<float>>();
        public List<List<float>> actionLog = new List<List<float>>();
    }
}
