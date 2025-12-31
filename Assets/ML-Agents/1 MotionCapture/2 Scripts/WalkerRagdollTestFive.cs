using System;
using Unity.MLAgentsExamples;
using UnityEngine;
//using UnityEditor;
using Newtonsoft.Json;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.IntegralTransforms;
using MathNet.Numerics;

//2.8记录：这个是最初的加载数据、预处理的

//更新，另开一个文件夹，保存所有脚部位置和com信息，用于控制系统的学习
//用于处理原始的数据集，把每个转化为关节角，并且消除根节点y轴导致的浮空
public class WalkerRagdollTestFive : MonoBehaviour
{

    /// <summary>
    /// 需要根据是否触地来改变颜色的object
    /// </summary>
    private GameObject[] meshChangeObject;
    public Material groundedMaterial;
    public Material unGroundedMaterial;

    public Transform cube;


    [Header("Body Parts")]
    public Transform hips;
    public Transform spine;
    public Transform thighL;
    public Transform shinL;
    public Transform thighR;
    public Transform shinR;
    public Transform FootMeshL;
    public Transform FootMeshR;

    [Header("Shin Collider")]
    public Collider shinLHeel;
    public Collider shinLToe;
    public Collider shinRHeel;
    public Collider shinRToe;

    /// <summary>
    /// 全部的json文件名字符串
    /// </summary>
    private List<string> fileList;

    [Header("Processing Information")]
    /// <summary>
    /// josn文件序号
    /// </summary>
    public int fileNum;
    /// <summary>
    /// 记录循环结束时位置的中间变量
    /// </summary>
    public Vector3 midStartPosition;
    JointDriveController m_JdController;
    /// <summary>
    /// 数据集中的walk数据（二维数组）,读取模式下是处理后毒，运行模式下是原始的
    /// </summary>
    private float[,] motionCaptureData;
    /// <summary>
    /// 原始的运动捕捉数据集列表，只包含位置、关节角，没有经过物理处理
    /// </summary>
    ImuData imuData;
    /// <summary>
    /// 处理后的运动捕捉数据集列表，包含需要模仿的序列的全部信息（包括角速度、com位置、足部位置等）
    /// </summary>
    ImuData2 imuData2;
    /// <summary>
    /// 记录读到了第几个文件，初始值是0
    /// </summary>
    RbVelocity rbVelocity;//test111111111
    private int i;
    /// <summary>
    /// 动作序列有几帧
    /// </summary>
    private int length;
    /// <summary>
    /// 数据集中行进方向和目标行进方向的夹角
    /// </summary>
    private float angleTowards;
    /// <summary>
    /// json文件的地址字符串
    /// </summary>
    string Dir;

    [Header("Physics Adjustment")]
    public int framerate;
    /// <summary>
    /// 最多使用几个物理帧优化一个动作
    /// </summary>
    public int physicsLimit;
    /// <summary>
    /// 当前属于第几个优化的物理帧
    /// </summary>
    public int adjustFrame;
    [Header("Playing Mode")]
    public bool showResult;
    /// <summary>
    /// 健康人大数据库为true，病人自测数据为false
    /// </summary>
    public bool ifHealth;
    /// <summary>
    /// 模拟中用到的关节，本实验中不包括上肢
    /// </summary>
    List<Transform> bodylistInUse;

    public void Start()
    {
        //Setup each body part，储存各种初始位置
        m_JdController = GetComponent<JointDriveController>();
        m_JdController.SetupBodyPart(hips);
        m_JdController.SetupBodyPart(spine);
        m_JdController.SetupBodyPart(thighL);
        m_JdController.SetupBodyPart(shinL);
        m_JdController.SetupBodyPart(thighR);
        m_JdController.SetupBodyPart(shinR);

        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
            bodyPart.rb.useGravity = false;
        }

        meshChangeObject = GameObject.FindGameObjectsWithTag("mesh");

        i = 0;
        adjustFrame = 0;
        fileNum = 0;

        midStartPosition = Vector3.zero;

        bodylistInUse = new List<Transform> { hips, thighL, shinL, thighR, shinR, spine };

        physicsLimit = 0;

        //界面参数调整，读文件
        if (showResult == true)
        {
            if (ifHealth)//包括健康人和病人，这个需要根据数据序号调整
                fileList = GetFiles("D:/UNITY/ml-agent/mlagent/Project/Assets/StreamingAssets/OutputHealthDataset/", ".json");
            else //展示90Hz病人数据
                fileList = GetFiles("D:/UNITY/ml-agent/mlagent/Project/Assets/StreamingAssets/OutputPatientDataset/", ".json");
            framerate = 90;
        }
        else
        {
            if (ifHealth)//mocap数据集，原始数据
            {
                fileList = GetFiles("D:/UNITY/ml-agent/mlagent/Project/Assets/StreamingAssets/MocapData/", ".json");
                framerate = 120;
            }
            else//noitom数据集，原始数据
            {
                fileList = GetFiles("D:/UNITY/ml-agent/mlagent/Project/Assets/StreamingAssets/NoitomData/", ".json");
                framerate = 90;
            }
        }
        Time.fixedDeltaTime = 1.0f / framerate;
    }

    public class ImuData
    {
        public float[,] walk;
        public float aveVelocity;
        public float stepLength;
        public float stepFrequency;
        public float[,] fai;
    }

    public class RbVelocity
    {
        public float[,] rbVelocity;
        public float[,] rbAngleVelocity;
        public float[,] datasetVelocity;
        public float[,] datasetAngleVelocity;
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

    public class CustomArray<T>
    {
        public T[] GetColumn(T[,] matrix, int columnNumber)
        {
            return Enumerable.Range(0, matrix.GetLength(0))
                    .Select(x => matrix[x, columnNumber])
                    .ToArray();
        }

        public T[] GetRow(T[,] matrix, int rowNumber)
        {
            return Enumerable.Range(0, matrix.GetLength(1))
                    .Select(x => matrix[rowNumber, x])
                    .ToArray();
        }
    }

    void FixedUpdate()
    {

        //接触地面的变色
        GroundTouchVisualize();

        //已处理完一个序列，重新读文件，初始化储存变量的数组
        if (i == 0 && adjustFrame == 0)
        {
            if (fileNum == fileList.Count()) { fileNum = 0; }//所有序列都处理完，就从头来
            Dir = fileList[fileNum];
            foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
            {
                bodyPart.rb.velocity = Vector3.zero;
                bodyPart.rb.angularVelocity = Vector3.zero;
            }
            //读取数据
            //Dir = Application.streamingAssetsPath + "/Json/walk24.json"; //调试
            if (showResult == true)
            {
                if (ifHealth)//从全都包含的那个文件夹里读，需要判断序号。<400则120Hz，否则90Hz
                {
                    int num = (int)float.Parse(Dir.Remove(Dir.Length - 5, 5).Remove(0, 81));
                    framerate = num < 400 ? 120 : 90;
                    Time.fixedDeltaTime = 1.0f / framerate;
                    Debug.Log("展示健康人全部数据（需要分情况） 序号为：" + num+" 对应framerate为："+ framerate);
                }

                imuData2 = JsonConvert.DeserializeObject<ImuData2>(File.ReadAllText(@Dir));
                motionCaptureData = imuData2.walk;
                length = motionCaptureData.Length / 27;
                rbVelocity = new RbVelocity();
                
                rbVelocity.rbVelocity = new float[length, imuData2.bodyPartVelocity.Length / length];
                rbVelocity.rbAngleVelocity = new float[length, imuData2.angularVelocity.Length / length];
                rbVelocity.datasetVelocity = imuData2.bodyPartVelocity;
                rbVelocity.datasetAngleVelocity = imuData2.angularVelocity;
            }
            else //需要物理处理
            {
                //Dir = Application.streamingAssetsPath + "/jsonTestFive/walk300.json"; //调试
                imuData = JsonConvert.DeserializeObject<ImuData>(File.ReadAllText(@Dir));//解析自原始数据
                motionCaptureData = imuData.walk;
                length = motionCaptureData.Length / 62;
                imuData2 = new ImuData2();
                imuData2.endPosition = new float[length, 9]; imuData2.footContact = new int[length, 2]; imuData2.comPosition = new float[length, 3];
                imuData2.angularVelocity = new float[length, 18]; imuData2.bodyPartVelocity = new float[length, 18];
                imuData2.walk = new float[length, 27];

            }


            Debug.Log("动作序列号: " + fileNum + " 长度: " + length + " 地址" + Dir);


            MotionDataControl(motionCaptureData);//摆出初始姿势

            RecordJointPostureToController();
            if (showResult == false)  ImuData2logPosition();//记录数据;

            }
    
        if (i == length && adjustFrame == physicsLimit)//完成一个运动序列cycle，继续前进或，写文件
        {
            if (showResult == true)//如果为了展示，循环播放
            {
                if (Mathf.Abs(hips.position.x - m_JdController.bodyPartsDict[hips].startingPos.x) < 10)//目标是十米
                {
                    //Debug.Log("i: "+i);
                    midStartPosition = new Vector3(hips.position.x - m_JdController.bodyPartsDict[hips].startingPos.x, 0, 0);
                }
                else
                {
                    fileNum++;
                    midStartPosition = Vector3.zero;
                    string Dir2;
                    if (ifHealth)
                        Dir2 = "D:/UNITY/ml-agent/mlagent/Project/Assets/StreamingAssets/RigidbodyTest/" + Dir.Remove(0, 77);
                    else
                        Dir2 = "D:/UNITY/ml-agent/mlagent/Project/Assets/StreamingAssets/RigidbodyTest/" + Dir.Remove(0, 78);

                    File.WriteAllText(Dir2, JsonConvert.SerializeObject(rbVelocity), new System.Text.UTF8Encoding(false));
                }
            }
            if (showResult == false)//如果是为了物理调整，保存到原文件中
            {
                fileNum++;
                midStartPosition = Vector3.zero;

                UpdateIniVelocity();//把第二帧速度填入第一帧，把新的hip高度填进去
                UpdateStepInformation();//改成直接把原始的填入
                FinalProcessing();
                Debug.Log("行走平均速度：" + imuData2.aveVelocity+" i: "+i);

                //储存
                string Dir2;
                if (ifHealth)
                    Dir2 = "D:/UNITY/ml-agent/mlagent/Project/Assets/StreamingAssets/OutputHealthDataset/" + Dir.Remove(0, 67);
                else //分两种情况，<460时是自己测的健康人的（90Hz）,存在JsonWithPTestFive；>=460时是病人数据，存在JsonWithPPatientTestFive
                {
                    int num = (int)float.Parse(Dir.Remove(0, 72).Remove(3,5));//只留序号
                    if(num<460)
                        Dir2 = "D:/UNITY/ml-agent/mlagent/Project/Assets/StreamingAssets/OutputHealthDataset/" + Dir.Remove(0, 68);
                    else
                        Dir2 = "D:/UNITY/ml-agent/mlagent/Project/Assets/StreamingAssets/OutputPatientDataset/" + Dir.Remove(0, 68);
                    Debug.Log( "处理数据模式 "+" num: " + num + " 储存文件： "+ Dir2);
                }


                File.WriteAllText(Dir2, JsonConvert.SerializeObject(imuData2), new System.Text.UTF8Encoding(false));
            }
            i = 0; adjustFrame = -1;

        }
        else if (adjustFrame == physicsLimit)//每帧完成物理调整后，记录hip高度信息，摆新动作
        {
            if (i > 0 && i < length)//画辅助线
            {
                Transform lineBody = hips;
                Debug.DrawLine(m_JdController.bodyPartsDict[lineBody].PrePosition, lineBody.position, Color.grey, 2);
                //lineBody = shinL;
                //Debug.DrawLine(m_JdController.bodyPartsDict[lineBody].PrePosition, lineBody.position, Color.blue, 2);
            }

            if (i > 0 && (length - i) < 1 && showResult == false) TrailOnOff(false); else TrailOnOff(true);
            

            MotionDataControl(motionCaptureData); //摆动作

            //这里需要加一个检测到地面距离的

            RecordJointVelocityToController();//记录相对上一帧的线速度和角速度 ,通用
            if (showResult == false) ImuData2logVolocity();//储存新数据集中速度信息。包括angularVelocity和bodyPartVelocity;
            RecordJointPostureToController();//更新所有jointcontroller中的pre数据，记录当前rotation和postion（世界坐标系）,通用
            if (showResult == false) ImuData2logPosition();//记录数据,在这里加入了hip下降和足部触地;

            //储存rb中的量 test11111
            if (showResult == true && midStartPosition != Vector3.zero) RbVelocityLog();

            i++;
            adjustFrame = -1;
        }
        adjustFrame += 1;


    }

    private float Hipmove()
    {
        Vector3 footLv1 = shinL.localToWorldMatrix.MultiplyPoint(new Vector3(0, 0.94f / 2 - 0.15f, 0) + shinL.GetComponent<CapsuleCollider>().center);
        Vector3 footLv2 = shinL.localToWorldMatrix.MultiplyPoint(new Vector3(0, -0.94f / 2 + 0.15f, 0) + shinL.GetComponent<CapsuleCollider>().center);
        Vector3 footRv1 = shinR.localToWorldMatrix.MultiplyPoint(new Vector3(0, 0.94f / 2 - 0.15f, 0) + shinR.GetComponent<CapsuleCollider>().center);
        Vector3 footRv2 = shinR.localToWorldMatrix.MultiplyPoint(new Vector3(0, -0.94f / 2 + 0.15f, 0) + shinR.GetComponent<CapsuleCollider>().center);
        float r = Vector3.Magnitude(shinL.localToWorldMatrix.MultiplyPoint(new Vector3(0, 0.15f, 0)) - shinL.localToWorldMatrix.MultiplyPoint(new Vector3(0, 0, 0)));
        //Debug.DrawLine(footRv1, footRv2, Color.red, 1); Debug.DrawLine(footRv1, footRv2 + new Vector3(r, 0, 0), Color.red, 1); Debug.DrawLine(footRv1 + new Vector3(r, 0, 0), footRv2, Color.red, 1);

        Vector3 toeLv1 = shinL.localToWorldMatrix.MultiplyPoint(new Vector3(shinLToe.GetComponent<CapsuleCollider>().height / 2 - shinLToe.GetComponent<CapsuleCollider>().radius, 0,0) + shinLToe.GetComponent<CapsuleCollider>().center);
        Vector3 toeLv2 = shinL.localToWorldMatrix.MultiplyPoint(new Vector3(-shinLToe.GetComponent<CapsuleCollider>().height / 2 + shinLToe.GetComponent<CapsuleCollider>().radius, 0, 0) + shinLToe.GetComponent<CapsuleCollider>().center);
        Vector3 toeRv1 = shinR.localToWorldMatrix.MultiplyPoint(new Vector3(shinRToe.GetComponent<CapsuleCollider>().height / 2 - shinRToe.GetComponent<CapsuleCollider>().radius, 0, 0) + shinRToe.GetComponent<CapsuleCollider>().center);
        Vector3 toeRv2 = shinR.localToWorldMatrix.MultiplyPoint(new Vector3(-shinRToe.GetComponent<CapsuleCollider>().height / 2 + shinRToe.GetComponent<CapsuleCollider>().radius, 0, 0) + shinRToe.GetComponent<CapsuleCollider>().center);
        float r0 = Vector3.Magnitude(shinL.localToWorldMatrix.MultiplyPoint(new Vector3( shinRToe.GetComponent<CapsuleCollider>().radius,0, 0)) - shinL.localToWorldMatrix.MultiplyPoint(new Vector3(0, 0, 0)));
        //Debug.DrawLine(toeLv1, toeLv2, Color.red, 1); Debug.DrawLine(toeRv1, toeRv2 + new Vector3(0, -r0, 0), Color.red, 1); Debug.DrawLine(toeRv1 + new Vector3(0, -r0, 0), toeRv2, Color.red, 1);

        //检测射线碰撞
        RaycastHit[] hit;
        List<float> rayDistancelist = new List<float>();
        List<Vector3> startPointlist = new List<Vector3>() { footLv2 - new Vector3(0, r, 0), footRv2 - new Vector3(0, r, 0), toeLv1 - new Vector3(0, r0, 0), toeLv2 - new Vector3(0, r0, 0), toeRv1 - new Vector3(0, r0, 0), toeRv2 - new Vector3(0, r0, 0) };

        foreach (Vector3 startPoint in startPointlist)
        {
            hit = Physics.RaycastAll(startPoint, Vector3.down, 10f, ~(1 << 0));//有碰撞物体

            for (int c = 0; c < hit.Length; c++)
            {
                if (hit[c].collider.gameObject.CompareTag("ground"))//碰撞物体为地面
                { rayDistancelist.Add(hit[c].distance); }
            }
        }
        //Debug.Log("rayDistancelist长度(至少为6)： " + rayDistancelist.Count);

        //画离地距离的debug line
        if (showResult == false)
        {
            //选出startPointlist中y最小的一个，只从那里画线
            Vector3 radStart = startPointlist.OrderByDescending(a => a.y).Last();
            Debug.DrawLine(radStart, radStart - new Vector3(0, rayDistancelist.Min(), 0), Color.blue, 0.01f);
        }

        //给左右脚接触赋值
        float minHeightLeft = Mathf.Min(footLv2.y - r, toeLv1.y - r0, toeLv2.y - r0);
        float minHeightRight = Mathf.Min(footRv2.y - r, toeRv1.y - r0, toeRv2.y - r0);
        imuData2.footContact[i, 0] = Mathf.Abs(minHeightLeft - rayDistancelist.Min()) < 0.02f ? 1 : 0;
        imuData2.footContact[i, 1] = Mathf.Abs(minHeightRight - rayDistancelist.Min()) < 0.02f ? 1 : 0;

        return rayDistancelist.Min();//返回最近的地面
    }



    private void RbVelocityLog()
    {
        int p = 0;
        foreach (var trans in bodylistInUse)
        {
            rbVelocity.rbVelocity[i, p] = cube.GetComponent<Rigidbody>().velocity.x;//test111
            rbVelocity.rbAngleVelocity[i, p++] = cube.GetComponent<Rigidbody>().angularVelocity.x;//test111

            rbVelocity.rbVelocity[i, p] = cube.GetComponent<Rigidbody>().velocity.y;//test111
            rbVelocity.rbAngleVelocity[i, p++] = cube.GetComponent<Rigidbody>().angularVelocity.y;//test111

            rbVelocity.rbVelocity[i, p] = cube.GetComponent<Rigidbody>().velocity.z;//test111
            rbVelocity.rbAngleVelocity[i, p++] = cube.GetComponent<Rigidbody>().angularVelocity.z;//test111
        }
    }

    private void UpdateIniVelocity()
    {
        imuData2.angularVelocity = WriteIni(imuData2.angularVelocity, 0, 1,1);
        imuData2.bodyPartVelocity = WriteIni(imuData2.bodyPartVelocity, 0, length - 1, length - 1);
        imuData2.bodyPartVelocity = WriteIni(imuData2.bodyPartVelocity, 1, length - 1, length - 1);
        imuData2.bodyPartVelocity = WriteIni(imuData2.bodyPartVelocity, 2, length - 1, length - 1);
        imuData2.bodyPartVelocity = WriteIni(imuData2.bodyPartVelocity, 3, length - 1, length - 1);
        imuData2.bodyPartVelocity = WriteIni(imuData2.bodyPartVelocity, 4, length - 1, length - 1);
        imuData2.bodyPartVelocity = WriteIni(imuData2.bodyPartVelocity, 3, 3, 4);
        imuData2.bodyPartVelocity = WriteIni(imuData2.bodyPartVelocity, 2, 2, 3);
        imuData2.bodyPartVelocity = WriteIni(imuData2.bodyPartVelocity, 1, 1, 2);
        imuData2.bodyPartVelocity = WriteIni(imuData2.bodyPartVelocity, 0, 0, 1);
    }


    private float[,] WriteIni(float[,] data, int u,int p, int q)
    {
        for (int y = 0; y < data.Length / length; y++)
        {
            //Debug.Log("before: data[u, y]"+ data[u, y]);
            data[u, y] = (data[p, y] + data[q, y]) / 2;
            //Debug.Log("After: data[u, y]" + data[p, y]);
        }
        return data;
    }

    private void TrailOnOff(bool trailOn)
    {
        GameObject[] trailChangeObject;
        trailChangeObject = GameObject.FindGameObjectsWithTag("trail");
        foreach (var obj in trailChangeObject)
        {
            obj.GetComponent<TrailRenderer>().emitting = trailOn;
        }
    }

    /// <summary>
    /// 求重心位置com
    /// </summary>
    /// <returns>全局坐标系下重心位置</returns>
    private Vector3 ComCalculate()
    {
        Vector3 com = Vector3.zero; float mass = 0;
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {
             com += m_JdController.bodyPartsDict[trans].rb.mass * trans.position;
             mass += m_JdController.bodyPartsDict[trans].rb.mass;

        }
        com /= mass;//重心位置
        return com;
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
            }
        }

    }

    private void ImuData2logVolocity()
    {
        int p = 0;
        foreach (var trans in bodylistInUse)
        {

            imuData2.bodyPartVelocity[i, p] = m_JdController.bodyPartsDict[trans].velocity.x;
            imuData2.angularVelocity[i, p++] = m_JdController.bodyPartsDict[trans].angularVelocity.x;

            imuData2.bodyPartVelocity[i, p] = m_JdController.bodyPartsDict[trans].velocity.y;
            imuData2.angularVelocity[i, p++] = m_JdController.bodyPartsDict[trans].angularVelocity.y;

            imuData2.bodyPartVelocity[i, p] = m_JdController.bodyPartsDict[trans].velocity.z;
            imuData2.angularVelocity[i, p++] = m_JdController.bodyPartsDict[trans].angularVelocity.z;

        }
    }


    /// <summary>
    /// 储存新数据集中位置信息。包括walk（用到的姿态），footContact，endPosition，comPosition四个。
    /// </summary>
    private void ImuData2logPosition()
    {
        // comPosition
        Vector3 com = ComCalculate()-hips.position;
        imuData2.comPosition[i, 0] = com.x; imuData2.comPosition[i, 1] = com.y; imuData2.comPosition[i, 2] = com.z;

        // endPosition
        List<Transform> bodyEnd = new List<Transform> { shinL, shinR, spine };
        int count = 0;
        foreach (var trans in bodyEnd)
        {
            Vector3 endPosition = trans.position-hips.position;//是相对根节点的位置，不是相对父节点的！
            imuData2.endPosition[i, count++] = endPosition.x;
            imuData2.endPosition[i, count++] = endPosition.y;
            imuData2.endPosition[i, count++] = endPosition.z;
        }

        // walk 
        imuData2.walk[i, 0] = hips.position.x - m_JdController.bodyPartsDict[hips].startingPos.x;
        imuData2.walk[i, 1] = hips.position.y - m_JdController.bodyPartsDict[hips].startingPos.y -Hipmove(); 
        //Debug.Log("Hipmove()"+Hipmove());
        imuData2.walk[i, 2] = hips.position.z - m_JdController.bodyPartsDict[hips].startingPos.z;
        count = 3;
        foreach (var trans in bodylistInUse)//q order: x y z w
        {
            Quaternion bodyAngle = trans == hips ? trans.rotation : trans.localRotation;
            imuData2.walk[i, count++] = bodyAngle.x; imuData2.walk[i, count++] = bodyAngle.y;
            imuData2.walk[i, count++] = bodyAngle.z; imuData2.walk[i, count++] = bodyAngle.w;
        }

    }


    /// <summary>
    /// 由四元数变化计算角速度，输出速度单位为rad/s
    /// </summary>
    /// <param name="preRoation">上一帧</param>
    /// <param name="thisRotation">当前的</param>
    /// <returns></returns>
    public Vector3 AngularVelocityCalculate(Quaternion preRoation, Quaternion thisRotation)
    {
        float angleInDegrees;
        Vector3 rotationAxis;
        //Quaternion myQuaternion = Quaternion.Inverse(preRoation) * thisRotation;
        Quaternion myQuaternion = thisRotation * Quaternion.Inverse(preRoation);

        myQuaternion.ToAngleAxis(out angleInDegrees, out rotationAxis);

        Vector3 angularDisplacement = rotationAxis * angleInDegrees * Mathf.Deg2Rad;
        Vector3 AngularVelocityThis = angularDisplacement / Time.fixedDeltaTime;


        return AngularVelocityThis;
    }

    /// <summary>
    /// 纠正旋转顺序。基于动作捕捉数据，求得绕根节点旋转四元数,旋转顺序XYZ
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
    /// 指定hip位置和8个关节的旋转角(载入数据集中的动作捕捉数据)，第i帧
    /// </summary>
    private void MotionDataControl(float[,] motionData )
    {
        if (showResult == false) MotionDataControlData1(motionData);//填入原始数据，角度信息
        if (showResult == true) MotionDataControlData2(motionData);//填入物理处理后的数据，四元数信息
        hips.position += midStartPosition;

        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.rb.velocity = Vector3.zero;
            bodyPart.rb.angularVelocity = Vector3.zero;
        }
    }

    private void MotionDataControlData1(float[,] motionData)//一共62维
    {

        if (ifHealth)//mocap数据集
        {
            angleTowards = Vector3.Angle(new Vector3(motionCaptureData[length - 1, 0], 0, motionCaptureData[length - 1, 2]), Vector3.right);
            if (motionCaptureData[length - 1, 2] < motionCaptureData[0, 2]) angleTowards += 180;

            //hips的position是相对于初始位置的
            hips.position = m_JdController.bodyPartsDict[hips].startingPos + Quaternion.AngleAxis(angleTowards, Vector3.up) * new Vector3(motionData[i, 0], motionData[i, 1], motionData[i, 2]);
            hips.rotation = m_JdController.bodyPartsDict[hips].startingRot * AngleRoatation(motionData[i, 3], -motionData[i, 4], -motionData[i, 5]);

            if (motionCaptureData[length - 1, 2] < motionCaptureData[0, 2]) hips.rotation *= Quaternion.AngleAxis(180, Vector3.up);

            thighR.localRotation = AngleRoatation(0, 0, 20) * AngleRoatation(motionData[i, 48], -motionData[i, 49], -motionData[i, 50]);
            shinR.localRotation = AngleRoatation(motionData[i, 51], 0, 0);
            thighL.localRotation = AngleRoatation(0, 0, -20) * AngleRoatation(motionData[i, 55], -motionData[i, 56], -motionData[i, 57]);
            shinL.localRotation = AngleRoatation(motionData[i, 58], 0, 0);
            spine.localRotation = AngleRoatation(-10, 0, 0) * AngleRoatation(motionData[i, 6], -motionData[i, 7], -motionData[i, 8]) * AngleRoatation(motionData[i, 9], -motionData[i, 10], -motionData[i, 11]);
        }
        else//自己的动作捕捉: 顺序{hips位置，spine， thighL, shinL, thighR, shinR}，一共23维 
        {
            angleTowards = Vector3.Angle(new Vector3(motionCaptureData[length - 1, 0], 0, motionCaptureData[length - 1, 2]), Vector3.right);

            hips.position = m_JdController.bodyPartsDict[hips].startingPos + Quaternion.AngleAxis(angleTowards, Vector3.up) * new Vector3(motionData[i, 0], motionData[i, 1], motionData[i, 2]);
            hips.localRotation = new Quaternion(motionData[i, 3], -motionData[i, 4], -motionData[i, 5], motionData[i, 6]);
            //Debug.Log("Vector: "+ new Vector3(motionCaptureData[length - 1, 0], 0, motionCaptureData[length - 1, 2]) + " angle: "+ angleTowards+" Vector Result: "+ Quaternion.AngleAxis(angleTowards, Vector3.up) * new Vector3(motionData[i, 0], motionData[i, 1], motionData[i, 2]));

            thighR.localRotation = Quaternion.Inverse(hips.localRotation) * new Quaternion(motionData[i, 15], -motionData[i, 16], -motionData[i, 17], motionData[i, 18]);
            thighL.localRotation = Quaternion.Inverse(hips.localRotation) * new Quaternion(motionData[i, 7], -motionData[i, 8], -motionData[i, 9], motionData[i, 10]);
            //spine.localRotation = new Quaternion(motionData[i, 3], -motionData[i, 4], -motionData[i, 5], motionData[i, 6]);
            shinL.localRotation = Quaternion.Inverse(thighL.localRotation) * new Quaternion(motionData[i, 11], -motionData[i, 12], -motionData[i, 13], motionData[i, 14]);
            shinR.localRotation = Quaternion.Inverse(thighR.localRotation) * new Quaternion(motionData[i, 19], -motionData[i, 20], -motionData[i, 21], motionData[i, 22]);

        }




    }

    private void MotionDataControlData2(float[,] motionData)//一共35维
    {
        hips.position = m_JdController.bodyPartsDict[hips].startingPos + new Vector3(motionData[i, 0], motionData[i, 1], motionData[i, 2]);
        int count = 3;
        foreach (var trans in bodylistInUse)//q order: x y z w
        {
            Quaternion bodyAngle = new Quaternion(motionData[i, count++], motionData[i, count++], motionData[i, count++], motionData[i, count++]);
            if (trans == hips) trans.rotation = bodyAngle;
            else trans.localRotation = bodyAngle;
            //用来给cube摆位置
            if (trans == shinL)
            {
                Vector3 position1 = trans.position + midStartPosition + new Vector3(1, 1, 1);
                cube.GetComponent<Rigidbody>().MovePosition(position1);
                cube.GetComponent<Rigidbody>().MoveRotation(trans.rotation);
            }

        }
        
    }


    public static List<string> GetFiles(string path, string suffix)
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


    /// <summary>
    /// 记录此刻全身姿态至JdController，包括rotation、postion（世界坐标系）和足部接触状态
    /// </summary>
    public void RecordJointPostureToController()
    {
        float height = 0;
        if (showResult == false) height = Hipmove();
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {

            //更新位置（全局）
            m_JdController.bodyPartsDict[trans].PrePosition = trans.position - new Vector3(0, height, 0);
            //更新角度（全局）
            m_JdController.bodyPartsDict[trans].PreRoation = trans.rotation;
         
        }

        m_JdController.bodyPartsDict[shinL].preContact = m_JdController.bodyPartsDict[shinL].groundContact.touchingGround;
        m_JdController.bodyPartsDict[shinR].preContact = m_JdController.bodyPartsDict[shinR].groundContact.touchingGround;

    }

    /// <summary>
    /// 更新每两步的步长和频率,改为直接用matlab数据，不在这里算了
    /// </summary>
    private void UpdateStepInformation()
    {

        imuData2.stepLength = imuData.stepLength;
        imuData2.stepFrequency = imuData.stepFrequency;
        imuData2.fai = imuData.fai;
        imuData2.aveVelocity = imuData.aveVelocity;

        Debug.Log("aveVelocity: " + imuData2.aveVelocity + " stepFrequency: " + imuData2.stepFrequency + " stepLength: " + imuData2.stepLength);

    }


    /// <summary>
    /// 更新bodypart中记录的线速度和角速度，用当前的和上一次RecordPosture()记录的算
    /// </summary>
    private void RecordJointVelocityToController()
    {
        float height = 0;
        if(showResult==false) height = Hipmove();
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {
           //线速度计算
           m_JdController.bodyPartsDict[trans].velocity = (trans.position + new Vector3(0, -height, 0) - m_JdController.bodyPartsDict[trans].PrePosition) / Time.fixedDeltaTime;

            //角速度计算
            m_JdController.bodyPartsDict[trans].angularVelocity = AngularVelocityCalculate(m_JdController.bodyPartsDict[trans].PreRoation, trans.rotation);
        }
    }

    /// <summary>
    /// 最后对数据进行滑动平均滤波
    /// </summary>
    void FinalProcessing()
    {
        //平滑数据
        imuData2.endPosition = Smooth2(imuData2.endPosition, 6);
        imuData2.comPosition = Smooth2(imuData2.comPosition, 6);
        imuData2.walk = Smooth2(imuData2.walk, 6);
        imuData2.angularVelocity = Smooth2(imuData2.angularVelocity, 6);
        imuData2.bodyPartVelocity = Smooth2(imuData2.bodyPartVelocity, 10);
        imuData2.bodyPartVelocity = Smooth2(imuData2.bodyPartVelocity, 6);
        imuData2.footContact = SmoothFoot(imuData2.footContact, 4);

    }


    /// <summary>
    /// 滑动平均滤波
    /// </summary>
    /// <param name="data"></param>
    /// <param name=""></param>
    /// <returns></returns>
    int[,] SmoothFoot(int[,] data, int N)
    {
        int size = length;
        float Sum1 = 0;
        for (int col = 0; col < data.Length / size; col++)
        {
            for (int j = 0; j < size; j++)
            {
                if (j < N / 2)
                {
                    for (int k = 0; k < N; k++)
                    {
                        Sum1 += (float)data[j + k, col];
                    }
                    data[j, col] = Sum1 / N > 0.5 ? 1 : 0;
                }
                else
                    if (j < size - N / 2)
                {
                    for (int k = 0; k < N / 2; k++)
                    {
                        Sum1 += (float)(data[j + k, col] + data[j - k, col]);
                    }
                    data[j, col] = Sum1 / N > 0.5 ? 1 : 0;
                }
                else
                {
                    for (int k = 0; k < size - j; k++)
                    {
                        Sum1 += (float)data[j + k, col];
                    }
                    for (int k = 0; k < (N - size + j); k++)
                    {
                        Sum1 += (float)data[j - k, col];
                    }
                    data[j, col] = Sum1 / N > 0.5 ? 1 : 0;
                }
                Sum1 = 0;
            }
        }
        return data;


    }

    /// <summary>
    /// 滑动平均滤波
    /// </summary>
    /// <param name="data"></param>
    /// <param name=""></param>
    /// <returns></returns>
    float[,] Smooth2(float[,] data, int N)
    {
        int size = length;
        float Sum1 = 0;
        for (int col = 0; col < data.Length / size; col++)
        {
            for (int j = 0; j < size; j++)
            {
                if (j < N / 2)
                {
                    for (int k = 0; k < N; k++)
                    {
                        Sum1 += data[j + k, col];
                    }
                    data[j, col] = Sum1 / N;
                }
                else
                    if (j < size - N / 2)
                {
                    for (int k = 0; k < N / 2; k++)
                    {
                        Sum1 += (data[j + k, col] + data[j - k, col]);
                    }
                    data[j, col] = Sum1 / N;
                }
                else
                {
                    for (int k = 0; k < size - j; k++)
                    {
                        Sum1 += data[j + k, col];
                    }
                    for (int k = 0; k < (N - size + j); k++)
                    {
                        Sum1 += data[j - k, col];
                    }
                    data[j, col] = Sum1 / N;
                }
                Sum1 = 0;
            }
        }
        return data;
    }
}
