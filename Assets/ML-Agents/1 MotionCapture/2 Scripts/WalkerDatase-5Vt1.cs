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

/// 用于训练人体控制器，计算速度角速度，输出目标角度。并保存所需的observation和action

/// 改动记录：error多除的去掉了；验证关节力、力矩为零所以不加；action去掉head；去掉上肢；条权重效果不大；reward归在-1-+1之间效果不大；
/// 幅值设置成硬范围不好，改为高斯；评价指标不能用占最大幅值的百分比，因为有的约等于0；应该stack=2，因为reward反映的是上一帧

/// 在04和05中，framerate是120，reward分别为相加和相乘，结果都是3.85左右。使用的超参数一直是学习率0.005,3层，512节点，stack=1
/// 在08中，framerate是120，关节reward改为高斯型；序列长度固定为随机抽取250个；hip的observation改为相对位置，stack=1；变化不大，3.9左右
/// 在09中，framerate是30，关节力和力矩去掉，shin的角度变化太小，所以全部关节都加上了下一帧绝对数值；关节reward高斯型；各个关节之间差别不大，暂时先不加系数（512/4），误差7.9
/// 在10中，framerate是30，堆叠2轮，其他不变（512/4），误差减小
/// 在11中，framerate是30，堆叠2轮，reward中加入了脚部位置，所有reward都改成高斯型----误差和10差不多，观察到脚部位置误差逐渐增大，其他转好
/// 在12中，其他不变，reward改为相乘（最后试一次）？好像错选了序列不等长那个，可能要重新试一下
/// 在13中，把上肢的影响都消除(GUI中质量改为e-5；换了初始位置；updateposture里全部保持初始位置)6.5度
/// 在15中，更改几何，加上质量，重新运行，别的不变,reward是相乘，6.7度，但脚部的太大了
/// 在17中，把脚部的容差改大一些

public class WalkerDataset : Agent
{

    public float framerate;

    /// <summary>
    /// 如果是，则录制当前的obsevation、action；注：只能在inference状态下用（即有网络的情况下），用全部的序列长度；训练时开启的话会改成用250个的长度
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

    public int velocityRange; //计算平均速度时取多少个点，默认100

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
    public StatsRecorder statsRecorder;//tensorboard输出
    private float errorHipPosition;//记录位置误差
    private float errorFootPosition;
    private float errorAllRotation;
    private string dirOut;

    //记录训练后inference内容的列表
    private List<List<float>> observationLogList;
    private List<List<float>> actionLogList;

    private float IniFootHeightLeft;
    private float IniFootHeightRight;
    private float IniHipHeight;

    protected override void Awake()
    {
        base.Awake();
        statsRecorder = Academy.Instance.StatsRecorder;//tensorboard输出
        dirOut = "D:/UNITY/ml-agent/mlagent/Project/Assets/ObserAction/";Debug.Log(dirOut);
        Application.targetFrameRate = 30;
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

        //得到动作捕捉数据文件名列表
        fileList = GetFiles("D:/UNITY/ml-agent/mlagent/Project/Assets/StreamingAssets/JsonWithP", ".json");
        fileList = fileList.OrderBy(p => Guid.NewGuid().ToString()).ToList();//打乱顺序
        fileNum = 0;

    }


    public override void OnEpisodeBegin()
    {
        // 按顺序加载动作捕捉数据
        fileNum++;
        if (fileNum == fileList.Count()) { fileNum = 0; }
        imuData = JsonConvert.DeserializeObject<ImuData>(File.ReadAllText(@fileList[fileNum]));


        //如果要看特定的一个，关闭flagjson，然后替换成：
        //string dirOne = Application.streamingAssetsPath + "/JsonWithP/walk73.json";
        //imuData = JsonConvert.DeserializeObject<ImuData>(File.ReadAllText(@dirOne));
        //Debug.Log("单独看："+ dirOne);

        //*****这里是新改动的。赋值，任取250个赋值给motionCaptureData***********************
        int cut = (int)Math.Ceiling(120 / framerate);//framerate是仿真时用的速度，按文献中的30Hz

        float[,] motionCaptureDataCut = new float[imuData.walk.Length/62/cut, 62];//差值后的全部序列
        float[,] imuFootLCut = new float[imuData.walk.Length / 62 / cut, 3];
        float[,] imuFootRCut = new float[imuData.walk.Length / 62 / cut, 3];
        float[,] imuComCut = new float[imuData.walk.Length / 62 / cut, 3];

        //插值，每4个保留一个数
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
            int ini = UnityEngine.Random.Range(1, motionCaptureDataCut.Length / 62 - 63);//开始位置
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
            for (int m = 0; m < 62; m++)//初始位置归零
            {
                motionCaptureData[m, 0] -= motionCaptureDataCut[ini, 0];  motionCaptureData[m, 2] -= motionCaptureDataCut[ini, 2];
                imuFootL[m, 0] -= motionCaptureDataCut[ini, 0]; imuFootL[m, 2] -= motionCaptureDataCut[ini, 2];
                imuFootR[m, 0] -= motionCaptureDataCut[ini, 0]; imuFootR[m, 2] -= motionCaptureDataCut[ini, 2];
            }
        }
        //****************************************************************************

        //初始化数据集相关参数。帧数i，长度length，计算身体转角
        i = 0;
        length = motionCaptureData.Length / 62;
        angleTowards = Vector3.Angle(new Vector3(motionCaptureData[length - 1, 0], 0, motionCaptureData[length - 1, 2]), Vector3.right);//数据集中，方向角需要转多少
        //初始化角色姿态
        UpdatePosture(i);//填入第0帧动作捕捉数据，生成初始的姿态+hip位置
        RecordPosture();//记录当前rotation和postion（世界坐标系）

        i++;

        rewardStep = 0;
        //初始化tensorboard记录
        errorHipPosition = 0;//位置误差
        errorAllRotation = 0;
        errorFootPosition = 0;

        foreach (var bp in m_JdController.bodyPartsDict.Values)
        { bp.rotationError = 0; }
        positionlist = new List<float>();

        //需要记录的话就创建一个空的
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
        //在读取下一组数据之前更新reward，暂时放在这里
        if (i >= 2)
        { rewardStep = UpdateReward(); }
        else { rewardStep = 0; }

        UpdatePosture(i);//更新全身姿态+hip位置
        positionlist.Add(hips.position.x);

        // 每个体段的rigidbody相关信息
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {//一共9个关节，每个有4+4+3+3+1=15observation,hip-1
            if (trans != head && trans != handL && trans != handR && trans != forearmL && trans != forearmR && trans != armR && trans != armL)//放弃左右手
            {
                //把这一帧和下一帧的localrotation加入observation
                m_JdController.bodyPartsDict[trans].NextLocalRoation = RotationCalculate(trans, i + 1);
                sensor.AddObservation(trans.localRotation);
                sensor.AddObservation(m_JdController.bodyPartsDict[trans].NextLocalRoation);
                sensor.AddObservation(Quaternion.FromToRotation(trans.localRotation.eulerAngles, m_JdController.bodyPartsDict[trans].NextLocalRoation.eulerAngles));

                //这帧线速度加入observation
                m_JdController.bodyPartsDict[trans].velocity = (trans.position - m_JdController.bodyPartsDict[trans].PrePosition) / (1.0f/framerate);
                sensor.AddObservation(m_JdController.bodyPartsDict[trans].velocity);

                //这帧角速度加入observation
                m_JdController.bodyPartsDict[trans].angularVelocity = AngularVelocityCalculate(m_JdController.bodyPartsDict[trans].PreRoation, trans.rotation);
                sensor.AddObservation(m_JdController.bodyPartsDict[trans].angularVelocity);

                //当前关节扭矩上限加入observation（hip的这个为空）
                if (trans != hips)
                { sensor.AddObservation(m_JdController.bodyPartsDict[trans].currentStrength / m_JdController.maxJointForceLimit); }
            }
        }
        
        //hip位置（当前和与下一帧的距离）共3+3=6个
        sensor.AddObservation(hips.position - m_JdController.bodyPartsDict[hips].startingPos);//相对初始的位置
        sensor.AddObservation(m_JdController.bodyPartsDict[hips].startingPos + Quaternion.AngleAxis(angleTowards, Vector3.up) * new Vector3(motionCaptureData[i + 1, 0], motionCaptureData[i + 1, 1], motionCaptureData[i + 1, 2]) - hips.position);
        
        //脚是否接触地面，共2个
        sensor.AddObservation(m_JdController.bodyPartsDict[footL].groundContact.touchingGround);
        sensor.AddObservation(m_JdController.bodyPartsDict[footR].groundContact.touchingGround);

        if (flagLogJson)
        {
            List<float> listA = new List<float>();

            //（hip离地高度-初始值）/最大高度, float，1
            listA.Add((FootHeight(hips) - IniHipHeight) / 0.01f);
            Debug.Log("（hip离地高度-初始值）/最大高度" + (FootHeight(hips) - IniHipHeight));

            //（脚离地高度-初始值）/最大高度，float，2
            listA.Add(Math.Max(0,FootHeight(footL) - IniFootHeightLeft) / 0.3f);
            listA.Add(Math.Max(0, FootHeight(footR) - IniFootHeightRight) / 0.3f);
            Debug.Log("（脚离地高度-初始值）/最大高度" + (Math.Max(0, FootHeight(footL) - IniFootHeightLeft) / 0.3f));

            //目标速度-一段时间的平均速度，float，1
            int init ;
            if (i > velocityRange) { init = i - velocityRange; } else { init = 0; }//init为计时的初始位置
            float averVelocity = (hips.position.x - positionlist[init]) / (1.0f / framerate * (i - init));
            float velGoal = (Quaternion.AngleAxis(angleTowards, Vector3.up) * new Vector3(motionCaptureData[i, 0]-motionCaptureData[init, 0], motionCaptureData[i, 1] - motionCaptureData[init, 1], motionCaptureData[i, 2] - motionCaptureData[init, 2])/(1.0f/framerate* (i-init))).x;
            //Debug.Log("目标速度" + velGoal);
            listA.Add(velGoal - averVelocity);
            //Debug.Log("目标速度-平均速度" + (velGoal - averVelocity));

            //目标速度，float,1
            listA.Add(velGoal);

            //身体朝向与目标方向之差，4
            Vector3 bodyForward = (spine.forward + chest.forward) / 2;
            Vector3 bodyForwardGoal = Vector3.right;
            Quaternion ro = Quaternion.FromToRotation(bodyForward, bodyForwardGoal);
            listA.Add(ro.x); listA.Add(ro.y); listA.Add(ro.z); listA.Add(ro.w);
            //Debug.Log("身体朝向与目标方向之差" + ro);
            //Debug.Log("bodyForward" + bodyForward + "  bodyForwardGoal" + bodyForwardGoal);

            //脚是否接触地面,注意要换回true/false,2
            if (m_JdController.bodyPartsDict[footL].groundContact.touchingGround == true) { listA.Add(1); } else { listA.Add(0); }
            if (m_JdController.bodyPartsDict[footR].groundContact.touchingGround == true) { listA.Add(1); } else { listA.Add(0); }
            
            //加入各关节信息
            var bpDict = m_JdController.bodyPartsDict;
            List < BodyPart >  bpList = new List<BodyPart>() { bpDict[chest],bpDict[spine],bpDict[thighL],bpDict[shinL],bpDict[footL],bpDict[thighR],bpDict[shinR],bpDict[footR]};
            
            //先加hips的,4+3+3=10

            //全局旋转四元数，4
            listA.Add(hips.rotation.x); listA.Add(hips.rotation.y); listA.Add(hips.rotation.z); listA.Add(hips.rotation.w);
            //Debug.Log("hip旋转" + hips.rotation);

            //线速度，3
            listA.Add(bpDict[hips].velocity.x); listA.Add(bpDict[hips].velocity.y); listA.Add(bpDict[hips].velocity.z);
            //Debug.Log("线速度" + bpDict[hips].velocity);

            //角速度，3
            listA.Add(bpDict[hips].angularVelocity.x); listA.Add(bpDict[hips].angularVelocity.y); listA.Add(bpDict[hips].angularVelocity.z);;
            //Debug.Log("角速度" + bpDict[hips].angularVelocity);

            foreach (var bp in bpList)
            {   //一共8个关节，每个有4+3+3+1=11

                //localrotation四元数，4
                var rot = bp.rb.GetComponent<Transform>().localRotation;
                listA.Add(rot.x); listA.Add(rot.y); listA.Add(rot.z); listA.Add(rot.w);
                //Debug.Log("localrotation四元数" + rot);

                //线速度,3
                listA.Add(bp.velocity.x); listA.Add(bp.velocity.y); listA.Add(bp.velocity.z);
                //Debug.Log("线速度" + bp.velocity);

                //角速度,3
                listA.Add(bp.angularVelocity.x); listA.Add(bp.angularVelocity.y); listA.Add(bp.angularVelocity.z);
                //Debug.Log("角速度" + bp.angularVelocity);

                //扭矩上限,1
                listA.Add(bp.currentStrength / m_JdController.maxJointForceLimit);
                //Debug.Log("扭矩上限" + bp.currentStrength / m_JdController.maxJointForceLimit);
            }
            
            observationLogList.Add(listA);
        }

        RecordPosture();//更新rotation和postion（世界坐标系），用于下一次迭代
        RigidControl();//设置刚体的当前速度和角速度（世界坐标系）

    }



    public override void OnActionReceived(ActionBuffers actions)
    {
        var bpDict = m_JdController.bodyPartsDict;
        var n = -1;

        //对一下旋转轴一不一致
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
            //Debug.Log("序列结束end: "+i + "  积累奖励: " + GetCumulativeReward());

            foreach (var bp in bpDict.Values)
            { bp.rotationError /= i - 2; }//求error的平均值

            //inference调试的时候用
            //Debug.Log("各关节角度误差: chest" + bpDict[chest].rotationError + " spine" + bpDict[spine].rotationError+" hip"+ bpDict[hips].rotationError);
            //Debug.Log("thighL" + bpDict[thighL].rotationError + " shinL" + bpDict[shinL].rotationError + " footL" + bpDict[footL].rotationError);
            //Debug.Log("thighR" + bpDict[thighR].rotationError + " shinR" + bpDict[shinR].rotationError + " footL" + bpDict[footR].rotationError);

            //tensorboard记录
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
            EndEpisode();//如果该序列已模拟完，结束该episode
        }
        i++;

        AddReward(rewardStep);

    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        //这个只为调试，真正录的时候要开inference模式
        var continuousActions = actionsOut.ContinuousActions;
        for (int n = 0; n < continuousActions.Length; n++)
        {
            continuousActions[n] = 0;
        }
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
    /// 在episode结束时，将当前的observation和action保存到json文件中（附加到二维list的最后）
    /// </summary>
    /// <param name="observationLogList">observation的二维list，第一维是step序号，第二维是数据</param>
    /// <param name="actionLogList">action的二维list，第一维位置需要和observation的一一对应</param>
    /// <param name="dirOut">输出的json文件地址+文件名</param>
    /// <param name="dataFileName">本episode对应的数据集json文件</param>
    public void SaveObsActJson(List<List<float>> observation, List<List<float>> action, string dirOut, string dataFileName)
    {
        dataFileName = dataFileName.Remove(0, 67); ;//去掉前面的文件名，只保留“walk5.json”这些
        JsonDataLog jsonDataLog = new JsonDataLog();//新建一个类,并且要挨个把其中的列表都初始化
        
        //加入此episode的数据
        jsonDataLog.observationLog = observation;
        jsonDataLog.actionLog = action; 
      
        //保存json
        File.WriteAllText(dirOut + dataFileName, JsonConvert.SerializeObject(jsonDataLog), System.Text.Encoding.UTF8);
    }



    /// <summary>
    /// 基于动作捕捉数据，求得绕根节点旋转四元数,旋转顺序XYZ
    /// </summary>
    /// <param name="angleX"></param>
    /// <param name="angleY"></param>
    /// <param name="angleZ"></param>
    /// <returns>输出的旋转四元数</returns>
    private Quaternion AngleRoatation(float angleX, float angleY, float angleZ)
    {
        Quaternion RotationQ = Quaternion.Euler(new Vector3(angleX, 0, 0)) * Quaternion.Euler(new Vector3(0, angleY, 0)) * Quaternion.Euler(new Vector3(0, 0, angleZ));

        return RotationQ;
    }

    /// <summary>
    /// 把计算得到的目标速度和角速度加入刚体rigid控制中
    /// </summary>
    public void RigidControl()
    {
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {
            if (trans != head && trans != handL && trans != handR && trans != forearmL && trans != forearmR && trans != armR && trans != armL)
            {
                //为rigid控制器加入各个变量
                m_JdController.bodyPartsDict[trans].rb.angularVelocity = m_JdController.bodyPartsDict[trans].angularVelocity;
                m_JdController.bodyPartsDict[trans].rb.velocity = m_JdController.bodyPartsDict[trans].velocity;

            }
        }
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
        Vector3 AngularVelocityThis = angularDisplacement / (1.0f/framerate);

        return AngularVelocityThis;
    }


    /// <summary>
    /// 由数据集中的指定一帧数据，和当前的trans，计算localRotation
    /// </summary>
    /// <param name=""></param>
    public Quaternion RotationCalculate(Transform trans, int frameNumber)
    {
        Quaternion localRotation = new Quaternion();
        //行进方向需要改的
        //float angleTowards = Vector3.Angle(new Vector3(motionCaptureData[length - 1, 0], 0, motionCaptureData[length - 1, 2]), Vector3.right);

        if (trans == hips) localRotation = m_JdController.bodyPartsDict[hips].startingRot * AngleRoatation(motionCaptureData[frameNumber, 3], -motionCaptureData[frameNumber, 4], -motionCaptureData[frameNumber, 5]);
        //spine包括lowerback和upperback
        else if (trans == spine) localRotation = AngleRoatation(-10, 0, 0) * AngleRoatation(motionCaptureData[frameNumber, 6], -motionCaptureData[frameNumber, 7], -motionCaptureData[frameNumber, 8]) * AngleRoatation(motionCaptureData[frameNumber, 9], -motionCaptureData[frameNumber, 10], -motionCaptureData[frameNumber, 11]);
        else if (trans == chest) localRotation = AngleRoatation(motionCaptureData[frameNumber, 12], -motionCaptureData[frameNumber, 13], -motionCaptureData[frameNumber, 14]);

        else if (trans == armR) localRotation = m_JdController.bodyPartsDict[armR].startingLocalRot;// m_JdController.bodyPartsDict[armR].startingLocalRot * AngleRoatation(-30, 0, 0) * AngleRoatation(0, 0, -90) * AngleRoatation(motionCaptureData[frameNumber, 26], motionCaptureData[frameNumber, 27], motionCaptureData[frameNumber, 28]) * AngleRoatation(0, 0, 90) * AngleRoatation(30, 0, 0);// * AngleRoatation(10, 10, 0); //x是让手网内转，y让胳膊往后摆，z贴身体
        else if (trans == forearmR) localRotation = m_JdController.bodyPartsDict[forearmR].startingLocalRot;//AngleRoatation(0, -motionCaptureData[frameNumber, 29], 0); //对应rradius
        else if (trans == handR) localRotation = m_JdController.bodyPartsDict[handR].startingLocalRot;//AngleRoatation(motionCaptureData[frameNumber, 30], 0, 0) * AngleRoatation(0, 0, -motionCaptureData[frameNumber, 31]) * AngleRoatation(0, -motionCaptureData[frameNumber, 32], 0); //对应rwrist+rhand
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
    /// 更新全身的姿态,填入第frame帧的动作捕捉数据
    /// </summary>
    /// <param name="frame"></param>
    public void UpdatePosture(int frame)
    {
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {
            if (trans != head && trans != hips)//其余为局部旋转
            {
                trans.localRotation = RotationCalculate(trans, frame);
            }
            else if (trans == hips)//hip的是全局旋转
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
    }

    /// <summary>
    /// 比较当前值和第i帧数据的相似度
    /// </summary>
    public float UpdateReward()
    {
        float hipPositionReward;
        float bodyRotationReward = 0;//身体各段的局部旋转角
        float walkLengthReward = 0;
        float footPositionReward = 0;
        float rewardTotal;
        float errorTotal = 0;

        //公共的变量，用来储存中间值
        Vector3 datasetPre;//来自数据集（上一帧理论值）
        Vector3 datasetNow;//来自数据集（理论值）
        Vector3 datasetSim;//来自物理仿真（实验值）
        float maxMagnitude;//最大变化值
        float DeltaMagnitude;//实际变化值

        //hipPositionReward--------------------------------------------------
        maxMagnitude = 0.01f;
        //datasetPre = Quaternion.AngleAxis(angleTowards, Vector3.up) * new Vector3(motionCaptureData[i-1, 0], motionCaptureData[i-1, 1], motionCaptureData[i-1, 2]);
        datasetNow = Quaternion.AngleAxis(angleTowards, Vector3.up) * new Vector3(motionCaptureData[i, 0], motionCaptureData[i, 1], motionCaptureData[i, 2]);
        datasetSim = hips.position - m_JdController.bodyPartsDict[hips].startingPos;

        DeltaMagnitude = Vector3.Distance(datasetSim, datasetNow)/maxMagnitude;
        hipPositionReward = (float)Math.Exp(-DeltaMagnitude * DeltaMagnitude);

        errorHipPosition += Vector3.Distance(datasetSim, datasetNow);//记录误差

        //Debug.LogWarning("datasetnow: " + datasetNow + "  datasetPre: " + datasetPre + "  sim: " + datasetSim);
        //Debug.LogWarning("变化值上限: " + maxMagnitude + "  实际的变化值: " + Vector3.Distance(datasetSim, datasetNow));

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
        int m = 0;//记录有几个关节
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

                //*****************这是新换的，改成文献中指数的形式
                DeltaMagnitude = DeltaAngle(datasetSim, datasetNow) / maxMagnitude;
                bodyRotationReward += (float)Math.Exp(-DeltaMagnitude * DeltaMagnitude);
                //************************************************************

                errorTotal += DeltaAngle(datasetSim, datasetNow) ;
                //Debug.Log("DeltaAngle(datasetPre, datasetNow) " + DeltaAngle(datasetPre, datasetNow));
                //Debug.Log("DeltaAngle(datasetSim, datasetNow) " + DeltaAngle(datasetSim, datasetNow));
                m_JdController.bodyPartsDict[trans].rotationError += DeltaAngle(datasetSim, datasetNow);

                m++;
            }
        }
        bodyRotationReward /= m;

        //用于tensorboard记录
        errorTotal /= m; 
        errorAllRotation += errorTotal;

        //walkLengthReward，如果双脚均离地，给一个小的惩罚
        if (!m_JdController.bodyPartsDict[footL].groundContact.touchingGround && !m_JdController.bodyPartsDict[footR].groundContact.touchingGround)
        {
            walkLengthReward = - 0.01f;
            //Debug.Log("双脚未触地：" + fileList[fileNum]);
        }

        //求总的reward，这里不确定相乘还是加权
        //rewardTotal = 0.1f*hipPositionReward + 0.1f * footPositionReward  + 0.8f * bodyRotationReward + walkLengthReward;
        rewardTotal = hipPositionReward * footPositionReward * bodyRotationReward + walkLengthReward;
        return rewardTotal;
   
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


    /// <summary>
    /// 读取到的json数据
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
    /// 根据条件获取相关后缀的文件  没有子文件的
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
            if (filename.EndsWith(suffix))//判断文件后缀，并获取指定格式的文件全路径增添至fileList
            {
                fileList.Add(filename);
            }
        }
        return fileList;
    }

}
