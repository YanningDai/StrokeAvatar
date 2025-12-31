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

//2.17记录：正式Imiatation模型，开始的时候模仿的是指定数据集中的一个序列；之后把训练好的模型继续训练，模仿的是患者的一个
//注：如果只设速度目标，就模仿一个序列，自动改变速度；如果还有步长目标，就选择最像的序列
//最后有时间的话，再加上病人会选择reward最高的那个来模仿

public class Imitation : Agent
{
    private float framerate;
    //private float framerateDataset;
    public Transform Target;

    private float iniHipHeight;
    /// <summary>
    /// 需要根据是否触地来改变颜色的object
    /// </summary>
    private GameObject[] meshChangeObject;
    public Material groundedMaterial;
    public Material unGroundedMaterial;

    /// <summary>
    /// 记录每个文件对应步长的字典，key是步长，value是文件名
    /// </summary>
    [HideInInspector] public Dictionary<float, string> fileStepLength;
    /// <summary>
    /// 全部的json文件名字符串
    /// </summary>
    private List<string> fileList;
    /// <summary>
    /// 当前所用的json文件地址
    /// </summary>
    private string jsonDir;

    [Header("User Input Goal")]
    /// <summary>
    /// 是否随机目标步长，训练时为true，inference时为false
    /// </summary>
    public bool randomizeWalkLengthEachEpisode;
    /// <summary>
    /// 是否随机目标速度，训练时为true，inference时为false
    /// </summary>
    public bool randomizeWalkSpeedEachEpisode;
    /// <summary>
    /// 用户输入的目标步幅（2步的，应该是0.7-2？根据数据集来定）
    /// </summary>
    public float lengthInput;
    /// <summary>
    /// 用户输入的目标速度,应该是（1-2）
    /// </summary>
    public float velocityInput;
    private float freGoal;
    private float lengthGoal;
    /// <summary>
    /// 实际目标速度
    /// </summary>
    private float velocityGoal;

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
    /// <summary>
    /// 模拟中用到的关节，本实验中不包括上肢
    /// </summary>
    List<Transform> bodylistInUse;

    private ImuData2 imuData2;
    [HideInInspector] static public bool showtext;
    public StatsRecorder statsRecorder;//tensorboard输出

    /// <summary>
    /// 相位参数，取值0-1
    /// </summary>
    //[DisplayOnly]
    public float fai;

    public int velocityRange; //计算平均速度时取多少个点，默认100
    private List<float> positionlist;
    private List<Vector3> comPositionlist;
    float averVelocity;

    /// <summary>
    /// 用于fai的计数，每一个循环结束都归零
    /// </summary>
    int i;
    /// <summary>
    /// 用于reward中dataset计数，需要插值所以用float
    /// </summary>
    float iDataset;
    int clipLengthDataset;
    int clipLengthImitation;

    [Header("Reference Body Parts")]
    public Transform hipsRef;
    public Transform spineRef;
    public Transform thighLRef;
    public Transform shinLRef;
    public Transform footLRef;
    public Transform thighRRef;
    public Transform shinRRef;
    public Transform footRRef;

    private Vector3 xPositionBeforeI;
    private Vector3 averComVelocity;
    protected override void Awake()
    {
        base.Awake();
        statsRecorder = Academy.Instance.StatsRecorder;//tensorboard输出
        fileList = GetFiles("D:/UNITY/ml-agent/mlagent/Project/Assets/StreamingAssets/JsonWithP/", ".json");

        meshChangeObject = GameObject.FindGameObjectsWithTag("mesh");
        framerate = 30;
        //framerateDataset = 120;

        bodylistInUse = new List<Transform> { hips, thighL, shinL, footL, thighR, shinR, footR, spine };

        //这四行测试用，固定序列且固定起点
        lengthInput = 1.5096f;
        velocityInput = 1.3828f;
        randomizeWalkLengthEachEpisode = false;
        randomizeWalkSpeedEachEpisode = false;
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

        //ref中的关节，不确定加不加
        m_JdController.SetupBodyPart(hipsRef);
        m_JdController.SetupBodyPart(spineRef);
        m_JdController.SetupBodyPart(thighLRef);
        m_JdController.SetupBodyPart(shinLRef);
        m_JdController.SetupBodyPart(footLRef);
        m_JdController.SetupBodyPart(thighRRef);
        m_JdController.SetupBodyPart(shinRRef);
        m_JdController.SetupBodyPart(footRRef);

        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        fileStepLength = new Dictionary<float, string>();
        LogStepLengthInJson(fileList);//记录每个文件的步长
        //iniHipHeight = FootHeight(hips);

        showtext = true;
    }



    public override void OnEpisodeBegin()
    {
        i = 0;
        showtext = true;
        //初始化角色位置
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values) { bodyPart.Reset(bodyPart); };

        //确定要模仿的文件
        lengthGoal =
            randomizeWalkLengthEachEpisode ? Random.Range(1.12f, 1.68f) : lengthInput;//应该放一个已知的，1.5096f
        velocityGoal =
            randomizeWalkSpeedEachEpisode ? Random.Range(0.3f, 2.0f) : velocityInput;//1.3828f
        freGoal = velocityGoal / lengthGoal;
        jsonDir = FindFileByStepLength(lengthGoal);//根据目标步长确定本轮所用的序列,返回字符串
        //Debug.Log("jsonDir: " + jsonDir + " lengthGoal: " + lengthGoal + " velocityGoal: " + velocityGoal);
        imuData2 = JsonConvert.DeserializeObject<ImuData2>(File.ReadAllText(@jsonDir));//得到本轮所用的参考目标
        //***************如果要模仿给定的一个序列，就在这里盖过去

        //随机数，取一帧作为初始姿态，计算帧数和起始点等参数
        clipLengthDataset = imuData2.walk.Length / 35;
        clipLengthImitation = Mathf.RoundToInt(framerate / freGoal);
        Debug.Log("clipLengthDataset: " + clipLengthDataset + " clipLengthImitation: " + clipLengthImitation+ " jsonDir: " + jsonDir);


        i = Random.Range(0, clipLengthImitation-1);//在imitation中的位置
        //i = 0;//测试用2
        fai = (float)i / clipLengthImitation;//实际上训练用的fai
        iDataset = fai * (clipLengthDataset-1);

        //记录速度相关，
        positionlist = new List<float>();
        comPositionlist = new List<Vector3>();
        averVelocity = 0;
        xPositionBeforeI = -new Vector3(keyFrameLinearLerp(imuData2.walk, 0, iDataset).x, 0, 0);

        //Debug.Log("i: " + i+ " fai: " + fai+ " iDataset: " + iDataset);
        //摆出初始姿态
        SetIniPosture(iDataset);

        //ref摆姿势
        MotionDataControlRef(imuData2, iDataset);

        RecordPostureFoot();
    }



    public override void CollectObservations(VectorSensor sensor)
    {
        if(positionlist.Count>1) showtext = false;//每次重来都显示
        GroundTouchVisualize();//接触地面的变色

        //相位参数，1
        sensor.AddObservation(fai);

        //目标步频，1
        sensor.AddObservation(freGoal);

        //hip离地高度, float，1
        sensor.AddObservation(FootHeight(hips));

        //速度误差，float，1
        int init;
        init = positionlist.Count > velocityRange ? positionlist.Count - velocityRange : 0;  //init为计时的初始位置
        averVelocity = positionlist.Count>1?Mathf.Abs(positionlist[positionlist.Count-1] - positionlist[init]) / (1.0f / framerate * (positionlist.Count- init-1)): m_JdController.bodyPartsDict[hips].rb.velocity.x;
        averComVelocity = positionlist.Count > 1 ? (comPositionlist[positionlist.Count - 1] - comPositionlist[init]) / (1.0f / framerate * (positionlist.Count - init - 1)) : m_JdController.bodyPartsDict[hips].rb.velocity;
        //sensor.AddObservation(velocityGoal - averVelocity);
        sensor.AddObservation(new Vector3(velocityGoal, 0, 0) - averComVelocity);
        sensor.AddObservation(velocityGoal - Vector3.Dot(averComVelocity, Vector3.right));

        //目标速度，float,1
        sensor.AddObservation(velocityGoal);

        //脚是否接触地面，bool，2
        sensor.AddObservation(m_JdController.bodyPartsDict[footL].groundContact.touchingGround);
        sensor.AddObservation(m_JdController.bodyPartsDict[footR].groundContact.touchingGround);

        //加入各关节信息

        //先加hips的,4+3+3=10
        //全局旋转四元数，quaternion,4
        sensor.AddObservation(hips.rotation);
        //线速度，Vector3，3
        sensor.AddObservation(m_JdController.bodyPartsDict[hips].rb.velocity);
        //角速度，Vector3，3
        sensor.AddObservation(m_JdController.bodyPartsDict[hips].rb.angularVelocity);
        //****这里不确定加不加根节点的位置


        List<Transform> bodylistInUse2 = new List<Transform> { hips, thighL, shinL, footL, thighR, shinR, footR, spine, chest, head };
        foreach (var trans in bodylistInUse2)
        {
            Unity.MLAgentsExamples.BodyPart bp = m_JdController.bodyPartsDict[trans];

            //一共7个关节，每个有4+3+3+1=11,11*7=77;三个肢体末端位置，3*3=9
            if (trans != hips)
            {
                //localrotation，quaternion, 4
                sensor.AddObservation(trans.localRotation);
                //线速度,Vector3，3//******这里不确定用自带的刚体还是自己算的！
                sensor.AddObservation(bp.rb.velocity);
                //sensor.AddObservation(bp.velocity);
                //角速度,Vector3，3
                sensor.AddObservation(bp.rb.angularVelocity);
                //sensor.AddObservation(bp.angularVelocity);
                //扭矩上限,float, 1
                sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
            }
            if (trans == footL || trans == footR || trans == spine)//肢体末端相对根节点位置
                sensor.AddObservation(trans.position - hips.position);
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
        bpDict[spine].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], continuousActions[++n]);

        bpDict[thighL].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], 0);
        bpDict[thighR].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], 0);
        bpDict[shinL].SetJointTargetRotation(continuousActions[++n], 0, 0);
        bpDict[shinR].SetJointTargetRotation(continuousActions[++n], 0, 0);
        bpDict[footR].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], continuousActions[++n]);
        bpDict[footL].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], continuousActions[++n]);
        bpDict[chest].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], continuousActions[++n]);
        bpDict[head].SetJointTargetRotation(continuousActions[++n], continuousActions[++n], continuousActions[++n]);

        //update joint strength settings
        bpDict[spine].SetJointStrength(continuousActions[++n]);
        bpDict[thighL].SetJointStrength(continuousActions[++n]);
        bpDict[shinL].SetJointStrength(continuousActions[++n]);
        bpDict[footL].SetJointStrength(continuousActions[++n]);
        bpDict[thighR].SetJointStrength(continuousActions[++n]);
        bpDict[shinR].SetJointStrength(continuousActions[++n]);
        bpDict[footR].SetJointStrength(continuousActions[++n]);
        bpDict[chest].SetJointStrength(continuousActions[++n]);
        bpDict[head].SetJointStrength(continuousActions[++n]);

        positionlist.Add(hips.position.x);//待改positionlist.Add(hipsRef.position.x);
        comPositionlist.Add(comCalculate());

        if (Math.Abs(hips.position.x - Target.position.x) < 1)//快到目标就停止
        {
            SetReward(1);
            Debug.Log("到达目标");
            EndEpisode();
        }

        TotalRewardCalculate(imuData2, iDataset);
        RecordPostureFoot();

        i++;//这个要在算完reward之后更新
        if (i == clipLengthImitation) i = 0; //重回fai=0的位置
        fai = (float)i / clipLengthImitation;
        iDataset = fai * clipLengthDataset;

        MotionDataControlRef(imuData2, iDataset);
    }


    public override void Heuristic(in ActionBuffers actionsOut)
    {

    }
    private void MotionDataControlRef(ImuData2 imuData2, float iDataset)//一共35维
    {
        //先借用这个做一下中间变量
        if (positionlist.Count == 0)
            m_JdController.bodyPartsDict[hipsRef].PrePosition = xPositionBeforeI;
        else if (i == 0)
            m_JdController.bodyPartsDict[hipsRef].PrePosition = new Vector3(hipsRef.position.x - m_JdController.bodyPartsDict[hipsRef].startingPos.x, 0, 0);

        hipsRef.position = m_JdController.bodyPartsDict[hipsRef].startingPos+ m_JdController.bodyPartsDict[hipsRef].PrePosition + keyFrameLinearLerp(imuData2.walk, 0, iDataset);

        //Debug.Log("positionlist.Count: " + positionlist.Count+ " clipLengthDataset:" + clipLengthDataset+ " Mathf.Floor((float)positionlist.Count / clipLengthDataset):" + Mathf.Floor((float)positionlist.Count / clipLengthDataset));
        //Debug.Log("imuData2.stepLength: " + imuData2.stepLength);

        List<Transform> bodylistInUseRef = new List<Transform> { hipsRef, thighLRef, shinLRef, footLRef, thighRRef, shinRRef, footRRef, spineRef };
        int count = 0;
        foreach (var trans in bodylistInUseRef)//q order: x y z w
        {
            Quaternion bodyAngle = keyFrameQLerp(imuData2.walk, 3 + count * 4, iDataset);
            if (trans == hipsRef) trans.rotation = bodyAngle;
            else trans.localRotation = bodyAngle;
            //这里不能指定刚体速度，会被用物理计算，差很多
            count++;
        }

        foreach (var trans in bodylistInUseRef)
        {
            trans.GetComponent<Rigidbody>().velocity = Vector3.zero;
            trans.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        }
    }

    /// <summary>
    /// 用于加奖励,给定需要的数据set
    /// </summary>
    public void TotalRewardCalculate(ImuData2 imuData2, float iDataset)
    {
        // 1: 任务完成水平（包括走地形、方向、速度）-------------------------------------------------------------------------
        // 1.1 hip高度与地形高度一致，保留,额外增加左右
        var delatHeight = (Mathf.Abs(FootHeight(hips) - FootHeight(hipsRef)));//hip距离地面高度应该是0.7-0.9之间
        var delatLateral = (hips.position.z-m_JdController.bodyPartsDict[hips].startingPos.z)-(hipsRef.position.z - m_JdController.bodyPartsDict[hipsRef].startingPos.z);
        float hipHeightReward = (float)Math.Exp(-40 * delatHeight * delatHeight) * (float)Math.Exp(-200 * delatLateral * delatLateral);
        //Debug.Log("FootHeight(hips): " + FootHeight(hips) + " FootHeight(hipsRef): " + FootHeight(hipsRef) + " delatLateral: " + delatLateral + " hipHeightReward: " + hipHeightReward);//已验证

        // 1.2 行走的总方向要朝正方向，hip末位置减初位置，三轴，暂时省略
        Vector3 direction = new Vector3(hips.position.x - m_JdController.bodyPartsDict[hips].startingPos.x, 0, hips.position.z - m_JdController.bodyPartsDict[hips].startingPos.z).normalized;
        float deltadirection = 1 - Mathf.Clamp( Vector3.Dot(direction.normalized, Vector3.right),0,1);
        float forwardReward = (float)Math.Exp(-deltadirection * deltadirection);
        //Debug.Log("direction: " + direction.normalized + " deltadirection: " + Vector3.Dot(direction.normalized, Vector3.right) + " forwardReward: "+ forwardReward);//已验证

        // 1.3 速度一致，保留
        float deltaV = velocityGoal - Vector3.Dot(averComVelocity, Vector3.right);
        float velocityReward = Mathf.Exp(-2.5f * deltaV * deltaV);
        //Debug.Log(" velocityGoal: " + velocityGoal + " averVelocity: " + averVelocity + " velocityReward: " + velocityReward);//已验证

        // 1.4 长度一致,暂时省略
        float deltaLength = ((hipsRef.position.x - m_JdController.bodyPartsDict[hipsRef].startingPos.x) - (hips.position.x - m_JdController.bodyPartsDict[hips].startingPos.x))/ (hipsRef.position.x - m_JdController.bodyPartsDict[hipsRef].startingPos.x+0.01f);
        float LengthReward = (float)Math.Exp(-deltaLength * deltaLength);
        //Debug.Log(" data length: " + (hipsRef.position.x - m_JdController.bodyPartsDict[hipsRef].startingPos.x) + " imitation length: " + (hips.position.x - m_JdController.bodyPartsDict[hips].startingPos.x) + " deltaLength: " + deltaLength + " LengthReward: " + LengthReward);//已验证
        //Debug.DrawLine(hipsRef.position, m_JdController.bodyPartsDict[hipsRef].startingPos, Color.grey, 2);
        //Debug.DrawLine(hips.position, m_JdController.bodyPartsDict[hips].startingPos, Color.grey, 2);

        //float goalReward = hipHeightReward * velocityReward * forwardReward * LengthReward*10;
        float goalReward = velocityReward * forwardReward * 0.5f + hipHeightReward * 0.5f;
        //Debug.Log("velocityGoal: " + velocityGoal + " comvelocity: "+ averComVelocity +" com Forward Velocity"+ Vector3.Dot(averComVelocity, Vector3.right) + " deltaV2: " + deltaV2);

        // 2: 行走水平（双脚触地、脚步不拖行、走得远）------------------------------------------------------------------------
        // 2.1 脚着地
        float bodyStableReward = 0;
        if (m_JdController.bodyPartsDict[footL].groundContact.touchingGround && !m_JdController.bodyPartsDict[footR].groundContact.touchingGround)
        { bodyStableReward = -0.005f; }
        // 2.2 不拖地
        float footDragPenal = FootDrag();//不着地时为零，拖动时为负值
        // 2.3 走得远
        float walklengthReward = 0.005f;

        float walkAbilityReward = bodyStableReward + footDragPenal + walklengthReward;
        //Debug.Log("bodyStableReward(未触地): " + bodyStableReward + " footDragPenal: " + footDragPenal + " walklengthReward: " + walklengthReward);


        // 3：模仿相似度------------------------------------------------------------------------
        // 目标的帧数为：iDataset

        // 3.1 重心位置、肢体末端相对hip的位置
        //重心位置
        float delta1 = Vector3.Magnitude(comCalculate() - hips.position - keyFrameLinearLerp(imuData2.comPosition, 0, iDataset));//正常值0.07-0.1估计
        float delta2 = Vector3.Magnitude(footL.position - hips.position - keyFrameLinearLerp(imuData2.endPosition, 0, iDataset));//正常值0.07
        float delta3 = Vector3.Magnitude(footR.position - hips.position - keyFrameLinearLerp(imuData2.endPosition, 3, iDataset));
        float delta4 = Vector3.Magnitude(spine.position - hips.position - keyFrameLinearLerp(imuData2.endPosition, 6, iDataset));//正常值0.01

        //Debug.Log("delta1: "+ Math.Exp(-delta1 * delta1) + " delta2: " + Math.Exp(-delta2 * delta2) + " delta3: " + Math.Exp(-delta3 * delta3) + " delta4: " + Math.Exp(-delta4 * delta4));
        float keypositionReward = (float)Math.Exp(-40 * (delta2 * delta2 + delta3 * delta3 + delta4 * delta4));

        float compositionReward = (float)Math.Exp(-10 * delta1 * delta1);
        //Debug.Log("keypositionReward: " + keypositionReward);


        // 3.2 各关节的旋转角、角速度、速度
        // bodylistInUse = new List<Transform> { hips, thighL, shinL, footL, thighR, shinR, footR, spine };
        List<Transform> bodylistInUse4 = new List<Transform> { hipsRef, thighLRef, shinLRef, footLRef, thighRRef, shinRRef, footRRef, spineRef };
        float keyAngleReward = 0;
        float keyAngleVelociyReward = 0;
        float keyVelocityReward = 0;
        int count = 0;
        foreach (var trans in bodylistInUse4)
        {
            Unity.MLAgentsExamples.BodyPart bp = m_JdController.bodyPartsDict[trans];

            //关节速度,都是相对于世界坐标系，3维
            float deltaKeyV = Vector3.Magnitude(bp.rb.velocity - keyFrameLinearLerp(imuData2.bodyPartVelocity, count*3, iDataset));
            keyVelocityReward += deltaKeyV * deltaKeyV;

            //关节角速度,都是相对于世界坐标系，3维
            float deltaKeyAngleV = Vector3.Magnitude(bp.rb.angularVelocity - keyFrameLinearLerp(imuData2.angularVelocity, count * 3, iDataset));
            keyAngleVelociyReward += deltaKeyAngleV * deltaKeyAngleV;

            //关节角度，4维
            Quaternion bodyAngle = trans == hipsRef ? trans.rotation : trans.localRotation;
            Quaternion datasetAngle = keyFrameQLerp(imuData2.walk, 3 + count * 4, iDataset);
            float deltaKeyAngle = Quaternion.Angle(bodyAngle, datasetAngle) *Mathf.Deg2Rad;
            keyAngleReward += deltaKeyAngle * deltaKeyAngle;

            //if (trans == hipsRef) Debug.Log("reward中 hipsRef bodyAngle: " + bodyAngle.eulerAngles + " datasetAngle: " + datasetAngle.eulerAngles + "  deltaKeyAngle: " + Quaternion.Angle(bodyAngle, datasetAngle));
            //if (trans == thighLRef) Debug.Log("thighLRef bodyAngle: " + bodyAngle.eulerAngles + " datasetAngle: " + datasetAngle.eulerAngles + "  deltaKeyAngle: " + Quaternion.Angle(bodyAngle, datasetAngle));
            //if (trans == shinLRef) Debug.Log("shinLRef bodyAngle: " + bodyAngle.eulerAngles + " datasetAngle: " + datasetAngle.eulerAngles + "  deltaKeyAngle: " + Quaternion.Angle(bodyAngle, datasetAngle));
            //if (trans == footLRef) Debug.Log("footLRef bodyAngle: " + bodyAngle.eulerAngles + " datasetAngle: " + datasetAngle.eulerAngles + "  deltaKeyAngle: " + Quaternion.Angle(bodyAngle, datasetAngle));
            //if (trans == thighRRef) Debug.Log("thighRRef bodyAngle: " + bodyAngle.eulerAngles + " datasetAngle: " + datasetAngle.eulerAngles + "  deltaKeyAngle: " + Quaternion.Angle(bodyAngle, datasetAngle));
            //if (trans == shinRRef) Debug.Log("shinRRef bodyAngle: " + bodyAngle.eulerAngles + " datasetAngle: " + datasetAngle.eulerAngles + "  deltaKeyAngle: " + Quaternion.Angle(bodyAngle, datasetAngle));
            //if (trans == footRRef) Debug.Log("footRRef bodyAngle: " + bodyAngle.eulerAngles + " datasetAngle: " + datasetAngle.eulerAngles + "  deltaKeyAngle: " + Quaternion.Angle(bodyAngle, datasetAngle));
            //if (trans == spineRef) Debug.Log("spineRef bodyAngle: " + bodyAngle.eulerAngles + " datasetAngle: " + datasetAngle.eulerAngles + "  deltaKeyAngle: " + Quaternion.Angle(bodyAngle, datasetAngle));

            //List<Transform> bodylistInUseRef = new List<Transform> { hipsRef, thighLRef, shinLRef, footLRef, thighRRef, shinRRef, footRRef, spineRef };
            //int count = 0;
            //foreach (var trans in bodylistInUseRef)//q order: x y z w
            //{
            //    Quaternion bodyAngle = keyFrameQLerp(imuData2.walk, 3 + count * 4, iDataset);
            //    if (trans == hipsRef) trans.rotation = bodyAngle;
            //    else trans.localRotation = bodyAngle;
            //    count++;
            //}

            count++;
        }
        //让上半身直立的
        float deltaKeyAngle1 = Quaternion.Angle(chest.rotation, m_JdController.bodyPartsDict[chest].startingRot) * Mathf.Deg2Rad;
        keyAngleReward += deltaKeyAngle1 * deltaKeyAngle1;
        float deltaKeyAngle2 = Quaternion.Angle(head.rotation, m_JdController.bodyPartsDict[head].startingRot) * Mathf.Deg2Rad;
        keyAngleReward += deltaKeyAngle2 * deltaKeyAngle2;

        keyVelocityReward = (float)Math.Exp(-keyVelocityReward);//先省略
        keyAngleVelociyReward = (float)Math.Exp(-0.1 * keyAngleVelociyReward);
        keyAngleReward = (float)Math.Exp(-2 * keyAngleReward);

        //Debug.Log("keyAngleReward: " + keyAngleReward + " keyAngleVelociyReward: " + keyAngleVelociyReward + "  keyVelocityReward: " + keyVelocityReward);

        float keyFrameReward = 0.65f * keyAngleReward + 0.1f * keyAngleVelociyReward + 0.15f * keypositionReward + 0.1f * compositionReward;

        float totalReward = 0.2f * goalReward + 0.8f * keyFrameReward + walkAbilityReward;
        //totalReward = keyAngleReward*0.2f + walklengthReward + hipHeightReward*0.5f + LengthReward*0.3f;//调试用3
        //Debug.Log("totalReward: "+ totalReward);

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
        statsRecorder.Add("3.2 key Angle Reward", keyAngleReward, StatAggregationMethod.Average);
        statsRecorder.Add("3.3 key Angle Velocity Reward", keyAngleVelociyReward, StatAggregationMethod.Average);
        statsRecorder.Add("3.4 key Velocity Reward", keyVelocityReward, StatAggregationMethod.Average);
    }

    /// <summary>
    /// 对三维向量内容做线性插值
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
    /// 对关节旋转做四元数差值
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

        float deltaLeft = Vector3.Distance(footL.position, m_JdController.bodyPartsDict[footL].PrePosition) / 0.2f;
        float deltaRight = Vector3.Distance(footR.position, m_JdController.bodyPartsDict[footR].PrePosition) / 0.2f;
        //Debug.Log("leftpre: "+ leftpre + " rightpre: " + rightpre + " leftnow: " + leftnow + " rightnow: " + rightnow);
        //Debug.Log("footLRef.position: " + footLRef.position + " [footLRef].PrePosition: " + m_JdController.bodyPartsDict[footLRef].PrePosition + " deltaLeft: " + Vector3.Distance(footLRef.position, m_JdController.bodyPartsDict[footLRef].PrePosition));
        //Debug.Log("footRRef.position: " + footRRef.position + " [footRRef].PrePosition: " + m_JdController.bodyPartsDict[footRRef].PrePosition + " deltaLeft: " + Vector3.Distance(footRRef.position, m_JdController.bodyPartsDict[footRRef].PrePosition));

        rewardDrag -= leftpre * leftnow * (1 - (float)Math.Exp(-deltaLeft * deltaLeft));
        rewardDrag -= rightpre * rightnow * (1 - (float)Math.Exp(-deltaRight * deltaRight));

        //Debug.Log("rewardleft: " + leftpre * leftnow * (1 - (float)Math.Exp(-deltaLeft * deltaLeft)) + " rewardright: " + rightpre * rightnow * (1 - (float)Math.Exp(-deltaRight * deltaRight)) );

        return rewardDrag;
    }

    /// <summary>
    /// 计算脚离地高度
    /// </summary>
    /// <param name="foot"></param>
    /// <returns></returns>
    public float FootHeight(Transform foot)
    {
        RaycastHit[] hit;//检测射线碰撞

        List<float> rayDistancelist = new List<float>(); ;

        hit = Physics.RaycastAll(foot.position, Vector3.down, 500f, ~(1 << 0));//有碰撞物体

        if (hit.Length == 0) { return 0; }
        else
        {
            for (int c = 0; c < hit.Length; c++)
            {
                if (hit[c].collider.gameObject.CompareTag("ground"))//碰撞物体为地面
                { rayDistancelist.Add(hit[c].distance); }
            }

            return rayDistancelist.Min();//返回最近的地面
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
        Vector3 AngularVelocityThis = angularDisplacement / Time.fixedDeltaTime;

        return AngularVelocityThis;
    }


    /// <summary>
    /// 记录全身姿态，包括当前rotation和postion（世界坐标系）
    /// </summary>
    public void RecordPostureFoot()
    {
        m_JdController.bodyPartsDict[footL].preContact = m_JdController.bodyPartsDict[footL].groundContact.touchingGround;
        m_JdController.bodyPartsDict[footR].preContact = m_JdController.bodyPartsDict[footR].groundContact.touchingGround;
        m_JdController.bodyPartsDict[footLRef].preContact = m_JdController.bodyPartsDict[footLRef].groundContact.touchingGround;
        m_JdController.bodyPartsDict[footRRef].preContact = m_JdController.bodyPartsDict[footRRef].groundContact.touchingGround;

        m_JdController.bodyPartsDict[footL].PrePosition = footL.position;
        m_JdController.bodyPartsDict[footR].PrePosition = footR.position;
        m_JdController.bodyPartsDict[footLRef].PrePosition = footLRef.position;
        m_JdController.bodyPartsDict[footRRef].PrePosition = footRRef.position;
    }

    /// <summary>
    /// 对所有tag是mesh的物体，进行触地检测，触地时变色
    /// </summary>
    public void GroundTouchVisualize()
    {

        foreach (var meshObject in meshChangeObject)
        {
            //查找父级物体
            Transform parentTransform = meshObject.transform.parent;
            //如果父级物体有关节属性
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
        /// 身体位置和姿态，只包括需要用到的几个体段
        /// </summary>
        public float[,] walk;
        /// <summary>
        /// 肢体末端位置，0-2 footL，3-5 footR，6-8 spine
        /// </summary>
        public float[,] endPosition;
        /// <summary>
        /// 足部和地面是否接触，0 footL，1 footR；接触为1，不接触为0
        /// </summary>
        public int[,] footContact;
        public float[,] comPosition;
        /// <summary>
        /// 角速度，只包括需要用到的几个体段
        /// </summary>
        public float[,] angularVelocity;
        public float[,] bodyPartVelocity;
        /// <summary>
        /// 这段时间的步速
        /// </summary>
        public float aveVelocity;
        public float stepLength;
        public float stepFrequency;
        public float[,] fai;
    }

    /// <summary>
    /// 填写字典fileStepLength = new Dictionary<float, string>()
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
    /// 根据目标步长确定本轮所用的序列,返回字符串
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
        var (minValue, minIndex) = lengthList_abs.Select((x, i) => (x, i)).Min();//找到步长相差最小的那一帧

        //Debug.Log("查找和给定步长最近的文件 minValue " + minValue + " minIndex " + minIndex + " lengthList_origin[minIndex] "+ lengthList_origin[minIndex]);
        jsonDir = fileStepLength[lengthList_origin[minIndex]];
        //Debug.Log("查找和给定步长最近的文件 minValue " + minValue + " minIndex" + minIndex + " jsonDir " + jsonDir);
        return jsonDir;
    }

        /// <summary>
    /// 每一帧开头，姿态的随机初始化
    /// </summary>
    public void SetIniPosture(float keyini)
    {
        //Debug.Log("m_JdController.bodyPartsDict[hips].startingPos" + m_JdController.bodyPartsDict[hips].startingPos+ "xPositionBeforeI" + xPositionBeforeI);
        hips.position = m_JdController.bodyPartsDict[hips].startingPos + xPositionBeforeI + keyFrameLinearLerp(imuData2.walk, 0, keyini);

        var motionData = imuData2.walk;
        int count = 0;
        //int countv = 0;
        //int counta = 0;
        foreach (var trans in bodylistInUse)//q order: x y z w
        {
            //姿态指定
            Quaternion bodyAngle = keyFrameQLerp(motionData, 3 + count * 4, keyini);
            if (trans == hips) trans.rotation = bodyAngle;
            else trans.localRotation = bodyAngle;
            count++;
            //刚体速度指定
            //m_JdController.bodyPartsDict[trans].rb.velocity = keyFrameLinearLerp(imuData2.bodyPartVelocity, countv*3, keyini);
            //m_JdController.bodyPartsDict[trans].rb.angularVelocity = keyFrameLinearLerp(imuData2.angularVelocity, counta*3, keyini);
            //countv++; counta++;
        }
    }
    /// <summary>
    /// 求重心位置com
    /// </summary>
    /// <returns>全局坐标系下重心位置</returns>
    private Vector3 comCalculate()
    {
        Vector3 com = Vector3.zero; float mass = 0;
        List<Transform> bodylistInUse3 = new List<Transform> { hips, thighL, shinL, footL, thighR, shinR, footR, spine, chest, head,armL, forearmL, handL, armR , forearmR, handR };

        foreach (var trans in bodylistInUse3)
        {
            com += m_JdController.bodyPartsDict[trans].rb.mass * trans.position;
            mass += m_JdController.bodyPartsDict[trans].rb.mass;
        }
        com /= mass;//重心位置
        return com;
    }
}

//public class DisplayOnly : PropertyAttribute
//{

//}
//[CustomPropertyDrawer(typeof(DisplayOnly))]
//public class ReadOnlyDrawer : PropertyDrawer
//{
//    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
//    {
//        return EditorGUI.GetPropertyHeight(property, label, true);
//    }
//    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
//    {
//        GUI.enabled = false;
//        EditorGUI.PropertyField(position, property, label, true);
//        GUI.enabled = true;
//    }
//}

