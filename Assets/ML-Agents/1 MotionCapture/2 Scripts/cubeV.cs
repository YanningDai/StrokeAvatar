using System;
using Unity.MLAgentsExamples;
using UnityEngine;
//using UnityEditor;
using Newtonsoft.Json;
using System.IO;
using System.Collections.Generic;
using System.Linq;

public class CubeV : MonoBehaviour
{
    public int i ;
    public int fileNum ;
    private string Dir;
    string Dir2;
    private ImuData2 imuData2;
    public int length;
    private Rigidbody rb;
    private Vector3 positionDataset;
    private Vector3 velocityDataset;
    private Vector3 angleVelocityDataset;
    private Vector3 startPosition;
    private Quaternion startRotation;
    private RbVelocity rbVelocity;
    private Vector3 midStartPosition;
    private Quaternion rotationDataset;
    private List<string> fileList;
    // Start is called before the first frame update
    void Start()
    {
        Time.fixedDeltaTime = 1.0f / 120;
        i = 0;
        fileNum = 0;
        //Dir = Application.streamingAssetsPath + "/JsonWithP/walk14.json"; //调试
        
        fileList = getFiles("D:/UNITY/ml-agent/mlagent/Project/Assets/StreamingAssets/JsonWithP/", ".json");
        rb = gameObject.GetComponent<Rigidbody>();
        startPosition = gameObject.transform.position;
        startRotation = gameObject.transform.rotation;
        rbVelocity = new RbVelocity();

        midStartPosition = Vector3.zero;
    }
    // Update is called once per frame
    void FixedUpdate()
    {
        if (i == 0 && midStartPosition == Vector3.zero)
        {
            Debug.Log("fileNum" + fileNum);
            if (fileNum == fileList.Count()) { fileNum = 0; }//所有序列都处理完，就从头来
            Dir = fileList[fileNum];
            imuData2 = JsonConvert.DeserializeObject<ImuData2>(File.ReadAllText(@Dir));

            length = imuData2.walk.Length / 35;
            rbVelocity.rbVelocity = new float[length, 3];
            rbVelocity.datasetVelocity = new float[length, 3];
            rbVelocity.rbAngleVelocity = new float[length, 3];
            rbVelocity.datasetAngleVelocity = new float[length, 3];
        }



        if (i > 1)
        {
            rbVelocity.rbVelocity[i - 1, 0] = rb.velocity.x; rbVelocity.rbVelocity[i - 1, 1] = rb.velocity.y; rbVelocity.rbVelocity[i - 1, 2] = rb.velocity.z;
            rbVelocity.datasetVelocity[i - 1, 0] = velocityDataset.x; rbVelocity.datasetVelocity[i - 1, 1] = velocityDataset.y; rbVelocity.datasetVelocity[i - 1, 2] = velocityDataset.z;
            rbVelocity.rbAngleVelocity[i - 1, 0] = rb.angularVelocity.x; rbVelocity.rbAngleVelocity[i - 1, 1] = rb.angularVelocity.y; rbVelocity.rbAngleVelocity[i - 1, 2] = rb.angularVelocity.z;
            rbVelocity.datasetAngleVelocity[i - 1, 0] = angleVelocityDataset.x; rbVelocity.datasetAngleVelocity[i - 1, 1] = angleVelocityDataset.y; rbVelocity.datasetAngleVelocity[i - 1, 2] = angleVelocityDataset.z;
        }

        // 1 测试速度

        // 移动到位置
        positionDataset = new Vector3(imuData2.walk[i, 0], imuData2.walk[i, 1], imuData2.walk[i, 2]);
        velocityDataset = new Vector3(imuData2.bodyPartVelocity[i, 0], imuData2.bodyPartVelocity[i, 1], imuData2.bodyPartVelocity[i, 2]);
        rb.MovePosition(startPosition + positionDataset+ midStartPosition);
        //Debug.Log("rb.velocity " + rb.velocity+ " dataset velcocity: "+ velocityDataset);


        // 2 测试角速度
        rotationDataset = new Quaternion(imuData2.walk[i, 7], imuData2.walk[i, 8], imuData2.walk[i, 9], imuData2.walk[i, 10]);
        angleVelocityDataset = new Vector3(imuData2.angularVelocity[i, 3], imuData2.angularVelocity[i, 4], imuData2.angularVelocity[i, 5]);
        rb.MoveRotation(startRotation* rotationDataset);
        //Debug.Log("rb.Angle velocity " + rb.angularVelocity + " dataset Angle velcocity: " + angleVelocityDataset);

        if (i == length - 1)
        {
            i = -1;
            if (Mathf.Abs(gameObject.transform.position.x - startPosition.x) < 10)//目标是十米
            {
                midStartPosition = new Vector3(gameObject.transform.position.x - startPosition.x, 0, 0);
            }
            else
            {
                midStartPosition = Vector3.zero;
                Dir2 = "D:/UNITY/ml-agent/mlagent/Project/Assets/StreamingAssets/Rigidbody/" + Dir.Remove(0, 67);
                File.WriteAllText(Dir2, JsonConvert.SerializeObject(rbVelocity), new System.Text.UTF8Encoding(false));
                fileNum++;
            }
        }

        i++;
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
        Quaternion myQuaternion = Quaternion.Inverse(preRoation) * thisRotation;

        myQuaternion.ToAngleAxis(out angleInDegrees, out rotationAxis);

        Vector3 angularDisplacement = rotationAxis * angleInDegrees * Mathf.Deg2Rad;
        Vector3 AngularVelocityThis = angularDisplacement / Time.fixedDeltaTime;

        return AngularVelocityThis;
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
    public class RbVelocity
    {
        public float[,] rbVelocity;
        public float[,] rbAngleVelocity;
        public float[,] datasetVelocity;
        public float[,] datasetAngleVelocity;
    }
    public static List<string> getFiles(string path, string suffix)
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
