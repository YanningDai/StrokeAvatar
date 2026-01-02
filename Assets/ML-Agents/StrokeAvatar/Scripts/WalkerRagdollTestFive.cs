using System;
using Unity.MLAgentsExamples;
using UnityEngine;
//using UnityEditor;
using Newtonsoft.Json;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using MathNet.Numerics.IntegralTransforms;
using MathNet.Numerics;


// Process raw data: converts to joint angles and removes root-height floating on the Y axis
public class WalkerRagdollTestFive : MonoBehaviour
{

    /// <summary>
    /// Objects that need to change color based on ground contact
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
    /// Full list of JSON file names
    /// </summary>
    private List<string> fileList;

    [Header("Processing Information")]
    /// <summary>
    /// JSON file index
    /// </summary>
    public int fileNum;
    /// <summary>
    /// Temporary variable to store hip offset at the end of a loop
    /// </summary>
    public Vector3 midStartPosition;
    JointDriveController m_JdController;
    private float[,] motionCaptureData;
    /// <summary>
    /// Raw motion-capture dataset list with positions and joint angles (no physics applied)
    /// </summary>
    ImuData imuData;
    /// <summary>
    /// Processed motion-capture dataset with full imitation info (angular velocity, COM position, foot position, etc.)
    /// </summary>
    ImuData2 imuData2;
    /// <summary>
    /// Index of the current file, starts at 0
    /// </summary>
    RbVelocity rbVelocity;//test111111111
    private int i;
    /// <summary>
    /// Number of frames in the motion sequence
    /// </summary>
    private int length;
    /// <summary>
    /// Angle between dataset forward direction and target heading
    /// </summary>
    private float angleTowards;
    /// <summary>
    /// JSON file path
    /// </summary>
    string Dir;

    [Header("Physics Adjustment")]
    public int framerate;
    /// <summary>
    /// Max number of physics frames used to refine one action
    /// </summary>
    public int physicsLimit;
    /// <summary>
    /// Current index of the refining physics frame
    /// </summary>
    public int adjustFrame;
    [Header("Playing Mode")]
    public bool showResult;
    /// <summary>
    /// True for healthy dataset, false for patient self-test data
    /// </summary>
    public bool ifHealth;
    /// <summary>
    /// Joints used in simulation; upper limbs excluded in this experiment
    /// </summary>
    List<Transform> bodylistInUse;

    public void Start()
    {
        // Setup each body part and cache initial transforms
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

        if (showResult == true)
        {
            if (ifHealth)// includes both healthy and patient datasets
            {
                string dir = Path.Combine(Application.streamingAssetsPath,"OutputHealthDataset");
                fileList = GetFiles(dir, ".json");
            }
            else // display 90Hz patient dataset
            {
                string dir = Path.Combine(Application.streamingAssetsPath,"OutputPatientDataset");
                fileList = GetFiles(dir, ".json");
            }
            framerate = 90;
        }
        else
        {
            if (ifHealth)// mocap dataset, raw data
            {
                string dir = Path.Combine(Application.streamingAssetsPath,"MocapData");
                fileList = GetFiles(dir, ".json");
                framerate = 120;
            }
            else// noitom dataset, raw data
            {
                string dir = Path.Combine(Application.streamingAssetsPath,"NoitomData");
                fileList = GetFiles(dir, ".json");
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
        /// Body positions and orientations for the used body segments
        /// </summary>
        public float[,] walk;
        /// <summary>
        /// End-effector positions; 0-2 footL, 3-5 footR, 6-8 spine
        /// </summary>
        public float[,] endPosition;
        /// <summary>
        /// Foot-ground contact; 0 footL, 1 footR; 1 for contact, 0 otherwise
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

        // Change mesh color based on ground contact
        GroundTouchVisualize();

        // Finished a sequence; read next file and initialize storage variables
        if (i == 0 && adjustFrame == 0)
        {
            if (fileNum == fileList.Count()) { fileNum = 0; }// restart from the first file after all are processed
            Dir = fileList[fileNum];
            foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
            {
                bodyPart.rb.velocity = Vector3.zero;
                bodyPart.rb.angularVelocity = Vector3.zero;
            }
            // Read data
            //Dir = Application.streamingAssetsPath + "/Json/walk24.json"; //debug
            if (showResult == true)
            {
                if (ifHealth)// load from combined healthy dataset; decide sample rate by index (<400 => 120Hz, otherwise 90Hz)
                {
                    int num = (int)float.Parse(Dir.Remove(Dir.Length - 5, 5).Remove(0, 81));
                    framerate = num < 400 ? 120 : 90;
                    Time.fixedDeltaTime = 1.0f / framerate;
                    Debug.Log("Showing healthy data (select rate) index: " + num + " framerate: " + framerate);
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
            else // physics processing needed
            {
                //Dir = Application.streamingAssetsPath + "/jsonTestFive/walk300.json"; //debug
                imuData = JsonConvert.DeserializeObject<ImuData>(File.ReadAllText(@Dir));// parsed from raw data
                motionCaptureData = imuData.walk;
                length = motionCaptureData.Length / 62;
                imuData2 = new ImuData2();
                imuData2.endPosition = new float[length, 9]; imuData2.footContact = new int[length, 2]; imuData2.comPosition = new float[length, 3];
                imuData2.angularVelocity = new float[length, 18]; imuData2.bodyPartVelocity = new float[length, 18];
                imuData2.walk = new float[length, 27];

            }


            Debug.Log("Motion clip index: " + fileNum + " length: " + length + " path: " + Dir);


            MotionDataControl(motionCaptureData);// pose initial frame

            RecordJointPostureToController();
            if (showResult == false)  ImuData2logPosition();// record data

            }
    
        if (i == length && adjustFrame == physicsLimit)// finished one motion cycle; advance or save
        {
            if (showResult == true)// playback loop for visualization
            {
                if (Mathf.Abs(hips.position.x - m_JdController.bodyPartsDict[hips].startingPos.x) < 10)// target distance is 10 meters
                {
                    //Debug.Log("i: "+i);
                    midStartPosition = new Vector3(hips.position.x - m_JdController.bodyPartsDict[hips].startingPos.x, 0, 0);
                }
                else
                {
                    fileNum++;
                    midStartPosition = Vector3.zero;
                    string Dir2 = Path.Combine(Application.streamingAssetsPath,"RigidbodyTest",Path.GetFileName(Dir));

                    File.WriteAllText(Dir2, JsonConvert.SerializeObject(rbVelocity), new System.Text.UTF8Encoding(false));
                }
            }
            if (showResult == false)// for physics tuning, save back to dataset
            {
                fileNum++;
                midStartPosition = Vector3.zero;

                UpdateIniVelocity();// copy second-frame velocity into first frame and update hip height
                UpdateStepInformation();// use original step information
                FinalProcessing();
                Debug.Log("Walking average speed: " + imuData2.aveVelocity+" i: "+i);

                string Dir2;
                if (ifHealth)
                    Dir2 = Path.Combine(Application.streamingAssetsPath,"OutputHealthDataset",Path.GetFileName(Dir));
                else // two cases: <460 is self-recorded healthy (90Hz) in OutputHealthDataset; >=460 is patient data in OutputPatientDataset
                {
                    int num = int.Parse(Regex.Match(Path.GetFileNameWithoutExtension(Dir),@"\d+").Value);
                    if(num<460)
                        Dir2 = Path.Combine(Application.streamingAssetsPath,"OutputHealthDataset",Path.GetFileName(Dir));
                    else
                        Dir2 = Path.Combine(Application.streamingAssetsPath,"OutputPatientDataset",Path.GetFileName(Dir));
                    Debug.Log( "Data preprocessing "+" num: " + num + " save: "+ Dir2);
                }


                File.WriteAllText(Dir2, JsonConvert.SerializeObject(imuData2), new System.Text.UTF8Encoding(false));
            }
            i = 0; adjustFrame = -1;

        }
        else if (adjustFrame == physicsLimit)// after each physics refinement frame, log hip height and set next pose
        {
            if (i > 0 && i < length)// draw helper lines
            {
                Transform lineBody = hips;
                Debug.DrawLine(m_JdController.bodyPartsDict[lineBody].PrePosition, lineBody.position, Color.grey, 2);
                //lineBody = shinL;
                //Debug.DrawLine(m_JdController.bodyPartsDict[lineBody].PrePosition, lineBody.position, Color.blue, 2);
            }

            if (i > 0 && (length - i) < 1 && showResult == false) TrailOnOff(false); else TrailOnOff(true);
            

            MotionDataControl(motionCaptureData); // set pose

            RecordJointVelocityToController();// record linear and angular velocity relative to previous frame
            if (showResult == false) ImuData2logVolocity();// store velocity info into new dataset (angularVelocity and bodyPartVelocity)
            RecordJointPostureToController();// update all joint controller pre data; record current rotation and position (world)
            if (showResult == false) ImuData2logPosition();// record position data including hip drop and foot contact

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

        // Raycast collision detection
        RaycastHit[] hit;
        List<float> rayDistancelist = new List<float>();
        List<Vector3> startPointlist = new List<Vector3>() { footLv2 - new Vector3(0, r, 0), footRv2 - new Vector3(0, r, 0), toeLv1 - new Vector3(0, r0, 0), toeLv2 - new Vector3(0, r0, 0), toeRv1 - new Vector3(0, r0, 0), toeRv2 - new Vector3(0, r0, 0) };

        foreach (Vector3 startPoint in startPointlist)
        {
            hit = Physics.RaycastAll(startPoint, Vector3.down, 10f, ~(1 << 0));//有碰撞物体

            for (int c = 0; c < hit.Length; c++)
            {
                if (hit[c].collider.gameObject.CompareTag("ground"))// collider is ground
                { rayDistancelist.Add(hit[c].distance); }
            }
        }
        //Debug.Log("rayDistancelist count (at least 6): " + rayDistancelist.Count);

        // Draw debug line for ground distance
        if (showResult == false)
        {
            // choose the lowest start point in startPointlist and draw from there
            Vector3 radStart = startPointlist.OrderByDescending(a => a.y).Last();
            Debug.DrawLine(radStart, radStart - new Vector3(0, rayDistancelist.Min(), 0), Color.blue, 0.01f);
        }

        // Assign foot contact values
        float minHeightLeft = Mathf.Min(footLv2.y - r, toeLv1.y - r0, toeLv2.y - r0);
        float minHeightRight = Mathf.Min(footRv2.y - r, toeRv1.y - r0, toeRv2.y - r0);
        imuData2.footContact[i, 0] = Mathf.Abs(minHeightLeft - rayDistancelist.Min()) < 0.02f ? 1 : 0;
        imuData2.footContact[i, 1] = Mathf.Abs(minHeightRight - rayDistancelist.Min()) < 0.02f ? 1 : 0;

        return rayDistancelist.Min();// return nearest ground distance
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
    /// Calculate center of mass
    /// </summary>
    /// <returns>Center of mass in world space</returns>
    private Vector3 ComCalculate()
    {
        Vector3 com = Vector3.zero; float mass = 0;
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {
             com += m_JdController.bodyPartsDict[trans].rb.mass * trans.position;
             mass += m_JdController.bodyPartsDict[trans].rb.mass;

        }
        com /= mass;// center of mass
        return com;
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
    /// Store position-related info into the new dataset: walk (used poses), footContact, endPosition, comPosition
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
            Vector3 endPosition = trans.position-hips.position;// relative to root, not parent
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
    /// Compute angular velocity from quaternion delta in rad/s
    /// </summary>
    /// <param name="preRoation">previous frame rotation</param>
    /// <param name="thisRotation">current frame rotation</param>
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
    /// Correct rotation order: compute root-relative quaternion with XYZ order from mocap angles
    /// </summary>
    /// <param name="angleX"></param>
    /// <param name="angleY"></param>
    /// <param name="angleZ"></param>
    /// <returns>root-relative quaternion</returns>
    private Quaternion AngleRoatation(float angleX, float angleY, float angleZ)
    {
        Quaternion RotationQ = Quaternion.Euler(new Vector3(angleX, 0, 0)) * Quaternion.Euler(new Vector3(0, angleY, 0)) * Quaternion.Euler(new Vector3(0, 0, angleZ));

        return RotationQ;
    }

    /// <summary>
    /// Set hip position and rotations for eight joints from dataset for frame i
    /// </summary>
    private void MotionDataControl(float[,] motionData )
    {
        if (showResult == false) MotionDataControlData1(motionData);// use raw data (angles)
        if (showResult == true) MotionDataControlData2(motionData);// use processed data (quaternions)
        hips.position += midStartPosition;

        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.rb.velocity = Vector3.zero;
            bodyPart.rb.angularVelocity = Vector3.zero;
        }
    }

    private void MotionDataControlData1(float[,] motionData)// total 62 dimensions
    {

        if (ifHealth)// mocap
        {
            angleTowards = Vector3.Angle(new Vector3(motionCaptureData[length - 1, 0], 0, motionCaptureData[length - 1, 2]), Vector3.right);
            if (motionCaptureData[length - 1, 2] < motionCaptureData[0, 2]) angleTowards += 180;

            // hips position is relative to initial position
            hips.position = m_JdController.bodyPartsDict[hips].startingPos + Quaternion.AngleAxis(angleTowards, Vector3.up) * new Vector3(motionData[i, 0], motionData[i, 1], motionData[i, 2]);
            hips.rotation = m_JdController.bodyPartsDict[hips].startingRot * AngleRoatation(motionData[i, 3], -motionData[i, 4], -motionData[i, 5]);

            if (motionCaptureData[length - 1, 2] < motionCaptureData[0, 2]) hips.rotation *= Quaternion.AngleAxis(180, Vector3.up);

            thighR.localRotation = AngleRoatation(0, 0, 20) * AngleRoatation(motionData[i, 48], -motionData[i, 49], -motionData[i, 50]);
            shinR.localRotation = AngleRoatation(motionData[i, 51], 0, 0);
            thighL.localRotation = AngleRoatation(0, 0, -20) * AngleRoatation(motionData[i, 55], -motionData[i, 56], -motionData[i, 57]);
            shinL.localRotation = AngleRoatation(motionData[i, 58], 0, 0);
            spine.localRotation = AngleRoatation(-10, 0, 0) * AngleRoatation(motionData[i, 6], -motionData[i, 7], -motionData[i, 8]) * AngleRoatation(motionData[i, 9], -motionData[i, 10], -motionData[i, 11]);
        }
        else// noitom: order {hips position, spine, thighL, shinL, thighR, shinR}, 23 dimensions total 
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

    private void MotionDataControlData2(float[,] motionData)// total 35 dimensions
    {
        hips.position = m_JdController.bodyPartsDict[hips].startingPos + new Vector3(motionData[i, 0], motionData[i, 1], motionData[i, 2]);
        int count = 3;
        foreach (var trans in bodylistInUse)//q order: x y z w
        {
            Quaternion bodyAngle = new Quaternion(motionData[i, count++], motionData[i, count++], motionData[i, count++], motionData[i, count++]);
            if (trans == hips) trans.rotation = bodyAngle;
            else trans.localRotation = bodyAngle;
            // used to position the cube for visualization
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
            if (filename.EndsWith(suffix))// filter by suffix and add full paths
            {
                fileList.Add(filename);
            }
        }
        return fileList;
    }


    /// <summary>
    /// Record current full-body posture to JdController including rotation, position (world), and foot contact state
    /// </summary>
    public void RecordJointPostureToController()
    {
        float height = 0;
        if (showResult == false) height = Hipmove();
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {

            // update position (world)
            m_JdController.bodyPartsDict[trans].PrePosition = trans.position - new Vector3(0, height, 0);
            // update rotation (world)
            m_JdController.bodyPartsDict[trans].PreRoation = trans.rotation;
         
        }

        m_JdController.bodyPartsDict[shinL].preContact = m_JdController.bodyPartsDict[shinL].groundContact.touchingGround;
        m_JdController.bodyPartsDict[shinR].preContact = m_JdController.bodyPartsDict[shinR].groundContact.touchingGround;

    }

    /// <summary>
    /// Update step length and frequency directly from MATLAB-processed data
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
    /// Update recorded linear and angular velocities using current and previous posture data
    /// </summary>
    private void RecordJointVelocityToController()
    {
        float height = 0;
        if(showResult==false) height = Hipmove();
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {
           // linear velocity
           m_JdController.bodyPartsDict[trans].velocity = (trans.position + new Vector3(0, -height, 0) - m_JdController.bodyPartsDict[trans].PrePosition) / Time.fixedDeltaTime;

            // angular velocity
            m_JdController.bodyPartsDict[trans].angularVelocity = AngularVelocityCalculate(m_JdController.bodyPartsDict[trans].PreRoation, trans.rotation);
        }
    }

    /// <summary>
    /// Apply moving-average smoothing to the dataset
    /// </summary>
    void FinalProcessing()
    {
        // smooth data
        imuData2.endPosition = Smooth2(imuData2.endPosition, 6);
        imuData2.comPosition = Smooth2(imuData2.comPosition, 6);
        imuData2.walk = Smooth2(imuData2.walk, 6);
        imuData2.angularVelocity = Smooth2(imuData2.angularVelocity, 6);
        imuData2.bodyPartVelocity = Smooth2(imuData2.bodyPartVelocity, 10);
        imuData2.bodyPartVelocity = Smooth2(imuData2.bodyPartVelocity, 6);
        imuData2.footContact = SmoothFoot(imuData2.footContact, 4);

    }


    /// <summary>
    /// Moving-average filter for foot contact
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
    /// Moving-average filter for float data
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
