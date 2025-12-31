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

/// 正式模型，分为录制demo和测试两部分（包括RL+BC+GAIL）
public class StrokeWalker : Agent
{

    private float framerate = 30;
    /// <summary>
    /// 如果是，则载入json文件中的observation和action，录制demo（始终站在原地，在heristic模式下用）
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
    public StatsRecorder statsRecorder;//tensorboard输出
    private string dirOut;
    private JsonDataLog jsonDataLog;

    //记录训练后inference内容的列表
    private List<List<float>> observationLogList;
    private List<List<float>> actionLogList;
    private int jsonN;

    private float IniFootHeightLeft;
    private float IniFootHeightRight;
    private float IniHipHeight;
    private float averVelocityHip;

    /// <summary>
    /// 目标速度，用户输入,应该是（1-2,0,0）左右
    /// </summary>
    public float targetVelocity;
    private float velGoal;//实际使用的速度
    /// <summary>
    /// 是否随机目标速度，训练时为true，inference时为false
    /// </summary>
    public bool randomizeWalkSpeedEachEpisode;

    private int step;//记录位于episode的第几步
    public int velocityRange; //计算平均速度时取多少个点，默认100
    private List<float> positionlist;
    float averVelocity;

    float dataV;//demo数据中的最后一帧的目标速度
    int keyini;int keyend;
    int KeyPhaseNum;//当前进行到自己的第几个，如1-27
    int KeyPhaseTotal;//一共用几帧，如27，数据集中是37
    int KeyPhase;//当前用数据集中第几帧，如30

    protected override void Awake()
    {
        base.Awake();
        statsRecorder = Academy.Instance.StatsRecorder;//tensorboard输出
        //一次只录制一组动作。如果需要更多，在这个脚本里改，加一个计数
        dirOut = "D:/UNITY/ml-agent/mlagent/Project/Assets/ObserAction/walk73.json";
        Application.targetFrameRate = 30;

        keyini = 6;keyend = 40;

    }

    public override void Initialize()
    {
        Time.fixedDeltaTime = 1.0f / framerate;//物理帧时长

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

        //读取json文件
        jsonDataLog = JsonConvert.DeserializeObject<JsonDataLog>(File.ReadAllText(@dirOut));
        observationLogList = jsonDataLog.observationLog;//这些都是按30hz保存的
        actionLogList = jsonDataLog.actionLog;
        dataV = observationLogList[observationLogList.Count - 1][4];//目标速度（对本demo是固定的）

    }


    public override void OnEpisodeBegin()
    {
        //初始化角色姿态
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        //实际使用的目标速度
        velGoal = randomizeWalkSpeedEachEpisode ? UnityEngine.Random.Range(0.3f, 2) : targetVelocity;
        velGoal = 1.5f;
        IniFootHeightLeft = FootHeight(footL);
        IniFootHeightRight = FootHeight(footR);
        IniHipHeight = FootHeight(hips);
        jsonN = 0;

        RecordPosture();//记录当前rotation和postion（世界坐标系）

        //记录速度相关
        positionlist = new List<float>();
        positionlist.Add(m_JdController.bodyPartsDict[hips].startingPos.x);
        averVelocity = 0;
        step = 1;
        KeyPhaseNum = 0;
        KeyPhaseTotal = (int)Math.Round(dataV / targetVelocity * (keyend - keyini + 1), 0);//数据集中是37对应于这个速度的几帧
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        positionlist.Add(hips.position.x);

        if (flagDemo)//录demo
        {
            int n = 0;
            Vector3 vectorTem; Quaternion qTem; float floatTem; bool boolTem;

            //（hip离地高度-初始值）/最大高度, float，1
            floatTem = observationLogList[jsonN][n++]; sensor.AddObservation(floatTem);
            //Debug.Log("（hip离地高度-初始值）/最大高度" + floatTem);

            //（脚离地高度-初始值）/最大高度，float，2
            floatTem = observationLogList[jsonN][n++]; sensor.AddObservation(floatTem);
            floatTem = observationLogList[jsonN][n++]; sensor.AddObservation(floatTem);
            //Debug.Log("（脚离地高度-初始值）/最大高度" + floatTem);

            //目标速度-一段时间的平均速度，float，1
            floatTem = observationLogList[jsonN][n++]; sensor.AddObservation(floatTem);
            //Debug.Log("（目标速度-平均速度）/目标速度" + floatTem);

            //目标速度，float,1
            floatTem = observationLogList[jsonN][n++]; sensor.AddObservation(floatTem);

            //身体朝向与目标方向之差，quaternion,4
            qTem = new Quaternion(observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++]);
            sensor.AddObservation(qTem);
            //Debug.Log("身体朝向与目标方向之差" + qTem);

            //脚是否接触地面，bool，2
            floatTem = observationLogList[jsonN][n++]; if (floatTem == 1) { boolTem = true; } else { boolTem = false; }
            sensor.AddObservation(boolTem);
            //Debug.Log("左脚" + boolTem);
            floatTem = observationLogList[jsonN][n++]; if (floatTem == 1) { boolTem = true; } else { boolTem = false; }
            sensor.AddObservation(boolTem);
            //Debug.Log("右脚" + boolTem);

            //加入各关节信息
            //先加hips的,4+3+3=10

            //全局旋转四元数，quaternion,4
            qTem = new Quaternion(observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++]);
            sensor.AddObservation(qTem);
            //Debug.Log("hip旋转" + qTem);

            //线速度，Vector3，3
            vectorTem = new Vector3(observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++]);
            sensor.AddObservation(vectorTem);
            //Debug.Log("线速度" + vectorTem);

            //角速度，Vector3，3
            vectorTem = new Vector3(observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++]);
            sensor.AddObservation(vectorTem);
            //Debug.Log("角速度" + vectorTem);

            for (int m = 0; m < 8; m++)
            {
                //localrotation四元数，4
                qTem = new Quaternion(observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++]);
                sensor.AddObservation(qTem);
                //Debug.Log("localrotation四元数" + qTem);

                //线速度,3
                vectorTem = new Vector3(observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++]);
                sensor.AddObservation(vectorTem);
                //Debug.Log("线速度" + vectorTem);

                //角速度,3
                vectorTem = new Vector3(observationLogList[jsonN][n++], observationLogList[jsonN][n++], observationLogList[jsonN][n++]);
                sensor.AddObservation(vectorTem);
                //Debug.Log("角速度" + vectorTem);

                //扭矩上限,1
                floatTem = observationLogList[jsonN][n++]; sensor.AddObservation(floatTem);
                //Debug.Log("扭矩上限" + floatTem);
            }
            //因为不setposture，就每一帧都归一次零，防止摔倒
            foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
            { bodyPart.Reset(bodyPart); }
            //一共109个
        }

        else//正式运行，共109
        {
            JointVelocityCalculate();

            //（hip离地高度-初始值）/最大高度, float，1
            sensor.AddObservation((FootHeight(hips) - IniHipHeight) / 0.01f);

            //（脚离地高度-初始值）/最大高度，float，2
            sensor.AddObservation(Math.Max(0, FootHeight(footL) - IniFootHeightLeft) / 0.3f);
            sensor.AddObservation(Math.Max(0, FootHeight(footR) - IniFootHeightRight) / 0.3f);

            // 目标速度-一段时间的平均速度，float，1
            int init;
            if (step > velocityRange) { init = step - velocityRange; } else { init = 0; }//init为计时的初始位置
            averVelocityHip = (hips.position.x - positionlist[init]) / (1.0f / framerate * (step - init));

            averVelocity = 0;int q = 0;
            foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
            {
                averVelocity += bodyPart.rb.velocity.x;
                q++;
            }
            averVelocity /= q;

            sensor.AddObservation(velGoal - averVelocityHip);

            //目标速度，float,1
            sensor.AddObservation(velGoal);

            //身体朝向与目标方向之差，quaternion,4
            Vector3 bodyForward = (spine.forward + chest.forward) / 2;
            Vector3 bodyForwardGoal = Vector3.right;
            sensor.AddObservation(Quaternion.FromToRotation(bodyForward, bodyForwardGoal));

            //脚是否接触地面，bool，2
            sensor.AddObservation(m_JdController.bodyPartsDict[footL].groundContact.touchingGround);
            sensor.AddObservation(m_JdController.bodyPartsDict[footR].groundContact.touchingGround);

            //加入各关节信息
            var bpDict = m_JdController.bodyPartsDict;
            List<BodyPart> bpList = new List<BodyPart>() { bpDict[chest], bpDict[spine], bpDict[thighL], bpDict[shinL], bpDict[footL], bpDict[thighR], bpDict[shinR], bpDict[footR] };

            //先加hips的,4+3+3=10
            //全局旋转四元数，quaternion,4
            sensor.AddObservation(hips.rotation);
            //线速度，Vector3，3
            sensor.AddObservation(bpDict[hips].velocity);
            //角速度，Vector3，3
            sensor.AddObservation(bpDict[hips].angularVelocity);

            foreach (var bp in bpList)
            {//一共8个关节，每个有4+3+3+1=11
                Transform trans = bp.rb.GetComponent<Transform>();
                //localrotation，quaternion, 4
                sensor.AddObservation(trans.localRotation);
                //线速度,Vector3，3
                sensor.AddObservation(bp.velocity);
                //角速度,Vector3，3
                sensor.AddObservation(bp.angularVelocity);
                //扭矩上限,float, 1
                sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);

            }
        }
    }

    /// <summary>
    /// action函数，这个对各个脚本都一致
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

        //在最后记录各种状态
        RecordPosture();


        if (Math.Abs(hips.position.x - Target.position.x) < 1)//快到目标就停止
        {
            SetReward(1);
            Debug.Log("到达目标");
            EndEpisode();
        }
        if (hips.position.y < bpDict[hips].startingPos.y - 2)//摔下去就停止
        {
            SetReward(-1);
            Debug.Log("摔下平台");
            EndEpisode(); }
        step++;
        
    }


    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        if (flagDemo)//填入action
        {
            for (int n = 0; n < continuousActions.Length; n++)
            {
                continuousActions[n] = actionLogList[jsonN][n];
            }
            jsonN++;

            if (jsonN == actionLogList.Count)
            {
                Debug.Log("demo录制结束 "+ jsonN);
                jsonN = 0;
                //录制过程好像没法结束EndEpisode();
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
    /// 用于加奖励
    /// </summary>
    public void TotalRewardCalculate()
    {
        float totalReward;
        float goalReward;
        //float walkAbilityReward;
        float keyFrameReward;

        float hipHeightReward;//hip高度与地形高度一致
        float towardsReward;//方向一致
        float velocityReward;//速度一致
        float bodyStableReward;//身体稳
        float footDragPenal;//不拖地
        float walklengthReward;
        float keyHipHightReward;
        float keyHipRotReward;
        float keyRotReward = 0;
        float positionReward = 0;

        // 1: 任务完成水平（包括走地形、方向、速度）-------------------------------------------------------------------------
        // 1.1 hip高度与地形高度一致
        var delatHeight = Mathf.Clamp(Mathf.Abs(FootHeight(hips) - IniHipHeight), 0.15f, 2)/0.2f;
        hipHeightReward = (float)Math.Exp(-delatHeight * delatHeight); 

        // 1.2 方向一致,用点乘衡量
        Vector3 bodyForward = (spine.forward + hips.forward + chest.forward) / 3;
        Vector3 bodyUpward = (spine.up + hips.up + chest.up) / 3;
        //towardsReward = ((Vector3.Dot(bodyForward, Vector3.right) + 1) * 0.5f)*((Vector3.Dot(bodyUpward, Vector3.up) + 1) * 0.5f);
        towardsReward = (float)Math.Exp(-bodyForward.y * bodyForward.y)* (float)Math.Exp(-bodyForward.z * bodyForward.z);
        towardsReward *= (float)Math.Exp(-bodyUpward.z * bodyUpward.z) * (float)Math.Exp(-bodyUpward.z * bodyUpward.z);

        // 1.3 速度一致，用误差占比------改为全身平均速度*hip的100个点的速度
        float deltaV = (velGoal - averVelocity)/0.8f; float deltaVHip = (velGoal - averVelocityHip) / 0.8f;
        velocityReward = (float)Math.Exp(-deltaV * deltaV)* (float)Math.Exp(-deltaVHip * deltaVHip);
        //Debug.Log("deltaV: "+ deltaV + " deltaVHip: " + deltaVHip);

        //1.4 位置一致
        float deltaP = (velGoal/framerate*step + m_JdController.bodyPartsDict[hips].startingPos.x-hips.position.x) / 0.2f;
        positionReward = (float)Math.Exp(-deltaP * deltaP);
        //Debug.Log(positionReward);
        //goalReward = hipHeightReward * (float)Math.Pow(towardsReward,3)* velocityReward;
        goalReward = 0.2f*hipHeightReward + 0.2f*towardsReward +0.2f*velocityReward+0.3f* positionReward;
        goalReward = 0.2f * hipHeightReward + 0.2f * towardsReward + 0.5f * positionReward;

        //Debug.Log("hipHeightReward: "+ hipHeightReward + " towardsReward: "+ towardsReward + " velocityReward: " + velocityReward);

        // 2: 行走水平（双脚触地、脚步不拖行、走得远）------------------------------------------------------------------------
        // 2.1 脚着地
        if (m_JdController.bodyPartsDict[footL].groundContact.touchingGround && !m_JdController.bodyPartsDict[footR].groundContact.touchingGround)
        { bodyStableReward = -0.005f; }
        else
        { bodyStableReward = 0;}   
        // 2.2 不拖地
        footDragPenal = FootDrag();//不着地时为零，拖动时为负值
        // 2.3 走得远
        walklengthReward = 0.005f;

        //walkAbilityReward = bodyStableReward + footDragPenal + walklengthReward;
        //Debug.Log("bodyStableReward(未触地): " + bodyStableReward + " footDragPenal: " + footDragPenal + " walklengthReward: " + walklengthReward);


        // 3：关键帧动作相似度------------------------------------------------------------------------
        KeyPhase = keyini + (int)Math.Round((double)(keyend - keyini + 1) / KeyPhaseTotal * KeyPhaseNum, 0);//当前用数据集中第几帧，如30
        //Debug.Log("KeyPhaseTotal转化后帧数" + KeyPhaseTotal + "  KeyPhase当前帧" + KeyPhase);

        // 3.1 hip高度+足部高度
        float datahip = observationLogList[KeyPhase][0]*0.01f+IniHipHeight;//demo中的离地高度
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

            Quaternion dataR = new Quaternion(observationLogList[KeyPhase][numBp], observationLogList[KeyPhase][numBp+1], observationLogList[KeyPhase][numBp+2], observationLogList[KeyPhase][numBp+3]);
            DeltaMagnitude = DeltaAngle(dataR.eulerAngles, trans.localRotation.eulerAngles) / 15;
            keyRotReward += (float)Math.Exp(-DeltaMagnitude * DeltaMagnitude);
            numBp += 11;
            //Debug.Log("数据集中旋转角" + dataR.eulerAngles + "当前旋转角" + trans.localRotation.eulerAngles + "  keyRotReward" + keyRotReward);

        }
        if (step == 1) { keyRotReward = 0; }
        keyFrameReward = (keyHipHightReward + keyHipRotReward + keyRotReward) /11; 

        KeyPhaseNum++;//当前进行到自己的第几个，如1-27，每次加一
        if (KeyPhaseNum == KeyPhaseTotal)//当前phase结束
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

        float deltaLeft = Vector3.Distance(footL.position, m_JdController.bodyPartsDict[footL].PrePosition)/0.5f;
        float deltaRight = Vector3.Distance(footR.position, m_JdController.bodyPartsDict[footR].PrePosition) /0.5f;
        
        rewardDrag -= leftpre* leftnow* (1-(float)Math.Exp(-deltaLeft * deltaLeft));
        rewardDrag -= rightpre * rightnow * (1 - (float)Math.Exp(-deltaRight * deltaRight));
        rewardDrag /= 2;
        return rewardDrag;
    }


    /// <summary>
    /// 更新bodypart中记录的线速度和角速度
    /// </summary>
    private void JointVelocityCalculate()
    {
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {//一共9个关节，因为都是全局，所以hip不用单独说
            if (trans != head && trans != handL && trans != handR && trans != forearmL && trans != forearmR && trans != armR && trans != armL)//放弃左右手
            {
                //线速度计算
                m_JdController.bodyPartsDict[trans].velocity = (trans.position - m_JdController.bodyPartsDict[trans].PrePosition) / Time.fixedDeltaTime;
                //角速度计算
                m_JdController.bodyPartsDict[trans].angularVelocity = AngularVelocityCalculate(m_JdController.bodyPartsDict[trans].PreRoation, trans.rotation);
            }
        }

        //在这里加一下上肢动作消除
        armR.localRotation = m_JdController.bodyPartsDict[armR].startingLocalRot;
        forearmR.localRotation = m_JdController.bodyPartsDict[forearmR].startingLocalRot;
        handR.localRotation = m_JdController.bodyPartsDict[handR].startingLocalRot;
        armL.localRotation = m_JdController.bodyPartsDict[armL].startingLocalRot;
        forearmL.localRotation = m_JdController.bodyPartsDict[forearmL].startingLocalRot;
        handL.localRotation = m_JdController.bodyPartsDict[handL].startingLocalRot;

    }


    /// <summary>
    /// 计算脚离地高度
    /// </summary>
    /// <param name="foot"></param>
    /// <returns></returns>
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
    /// 由四元数变化计算角速度，输出速度单位为rad/s
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
    /// 记录全身姿态，包括当前rotation和postion（世界坐标系）
    /// </summary>
    public void RecordPosture()
    {
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {
            if (trans != head)
            {
                //更新位置（全局）
                m_JdController.bodyPartsDict[trans].PrePosition = trans.position;

                //更新角度（全局）
                m_JdController.bodyPartsDict[trans].PreRoation = trans.rotation;
            }
        }

        m_JdController.bodyPartsDict[footL].preContact = m_JdController.bodyPartsDict[footL].groundContact.touchingGround;
        m_JdController.bodyPartsDict[footR].preContact = m_JdController.bodyPartsDict[footR].groundContact.touchingGround;

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



    public class JsonDataLog
    {
        public List<List<float>> observationLog = new List<List<float>>();
        public List<List<float>> actionLog = new List<List<float>>();
    }


}

