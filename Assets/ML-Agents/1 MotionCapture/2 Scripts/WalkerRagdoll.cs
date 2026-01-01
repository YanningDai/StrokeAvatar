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

// Initial data loading and preprocessing (currently unused)

public class WalkerRagdoll : MonoBehaviour
{

    /// <summary>
    /// Objects that change color based on ground contact
    /// </summary>
    private GameObject[] meshChangeObject;
    public Material groundedMaterial;
    public Material unGroundedMaterial;

    public Transform cube;


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
    public Transform FootMeshL;
    public Transform FootMeshR;

    /// <summary>
    /// List of all JSON file names
    /// </summary>
    private List<string> fileList;
    /// <summary>
    /// JSON file index
    /// </summary>
    public int fileNum;

    /// <summary>
    /// Temporary variable for position at the end of a loop
    /// </summary>
    public Vector3 midStartPosition;
    JointDriveController m_JdController;
    /// <summary>
    /// Walk data from the dataset (2D array); processed when reading, raw when running
    /// </summary>
    private float[,] motionCaptureData;
    /// <summary>
    /// Raw motion-capture dataset (positions and joint angles only, no physics processing)
    /// </summary>
    ImuData imuData;
    /// <summary>
    /// Processed motion-capture dataset with full imitation info (angular velocity, COM, foot positions, etc.)
    /// </summary>
    ImuData2 imuData2;
    /// <summary>
    /// Tracks which file has been read; starts at 0
    /// </summary>
    RbVelocity rbVelocity;//test111111111
    private int i;
    /// <summary>
    /// Number of frames in the motion sequence
    /// </summary>
    private int length;
    /// <summary>
    /// Angle between dataset travel direction and target direction
    /// </summary>
    private float angleTowards;
    /// <summary>
    /// JSON file path
    /// </summary>
    string Dir;

    [Header("Physics Adjustment")]
    public int framerate;
    /// <summary>
    /// Maximum physics frames to refine one motion
    /// </summary>
    public int physicsLimit;
    /// <summary>
    /// Current physics frame index for optimization
    /// </summary>
    public int adjustFrame;
    [Header("Playing Mode")]
    public bool showResult;

    /// <summary>
    /// Joints used in the simulation; upper limbs excluded for this experiment
    /// </summary>
    List<Transform> bodylistInUse;

    public void Start()
    {
        // Setup each body part and cache initial poses
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
            bodyPart.rb.useGravity = false;
        }

        meshChangeObject = GameObject.FindGameObjectsWithTag("mesh");

        i = 0;
        adjustFrame = 0;
        fileNum = 0;

        midStartPosition = Vector3.zero;

        Time.fixedDeltaTime = 1.0f / framerate;

        bodylistInUse = new List<Transform> { hips, thighL, shinL, footL, thighR, shinR, footR, spine };

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
        /// Body positions/rotations for required body segments only
        /// </summary>
        public float[,] walk;
        /// <summary>
        /// End-effector positions; 0-2 footL, 3-5 footR, 6-8 spine
        /// </summary>
        public float[,] endPosition;
        /// <summary>
        /// Foot-ground contact; 0 footL, 1 footR; 1=contact, 0=no contact
        /// </summary>
        public int[,] footContact;
        public float[,] comPosition;
        /// <summary>
        /// Angular velocity for required body segments only
        /// </summary>
        public float[,] angularVelocity;
        public float[,] bodyPartVelocity;
        /// <summary>
        /// Average gait speed over this segment
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

        // UI parameter tuning and file loading
        Time.fixedDeltaTime = 1.0f / framerate;
        if (showResult == true)
        {
            fileList = GetFiles(Path.Combine(Application.streamingAssetsPath,"JsonWithP"), ".json");
            physicsLimit = 0;
        }
        else
            fileList = GetFiles(Path.Combine(Application.streamingAssetsPath,"json"), ".json");

        GroundTouchVisualize();

        // Log rigidbody values (test)
        if (showResult == true && midStartPosition != Vector3.zero) RbVelocityLog();

        // After finishing a sequence, read next file and initialize buffers
        if (i == 0 && adjustFrame == 0)
        {
            if (fileNum == fileList.Count()) { fileNum = 0; }// restart after finishing all sequences
            Dir = fileList[fileNum];
            foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
            {
                bodyPart.rb.velocity = Vector3.zero;
                bodyPart.rb.angularVelocity = Vector3.zero;
            }
            // Read data
            //Dir = Application.streamingAssetsPath + "/Json/walk24.json"; // debug
            if (showResult == true)
            {
                imuData2 = JsonConvert.DeserializeObject<ImuData2>(File.ReadAllText(@Dir));
                motionCaptureData = imuData2.walk;
                length = motionCaptureData.Length / 35;
                rbVelocity = new RbVelocity();
                //test111111
                rbVelocity.rbVelocity = new float[length, imuData2.bodyPartVelocity.Length / length];
                rbVelocity.rbAngleVelocity = new float[length, imuData2.angularVelocity.Length / length];
                rbVelocity.datasetVelocity = imuData2.bodyPartVelocity;
                rbVelocity.datasetAngleVelocity = imuData2.angularVelocity;
            }
            else // needs physics refinement
            {
                //Dir = Application.streamingAssetsPath + "/Json/walk62.json"; // debug
                imuData = JsonConvert.DeserializeObject<ImuData>(File.ReadAllText(@Dir));// parsed from raw data
                motionCaptureData = imuData.walk;
                length = motionCaptureData.Length / 62;
                imuData2 = new ImuData2();
                imuData2.endPosition = new float[length, 9]; imuData2.footContact = new int[length, 2]; imuData2.comPosition = new float[length, 3];
                imuData2.angularVelocity = new float[length, 24]; imuData2.bodyPartVelocity = new float[length, 24];
                imuData2.walk = new float[length, 35];
                angleTowards = Vector3.Angle(new Vector3(motionCaptureData[length - 1, 0], 0, motionCaptureData[length - 1, 2]), Vector3.right);
                if (motionCaptureData[length - 1, 2] < motionCaptureData[0, 2]) angleTowards += 180;
            }


            Debug.Log("Sequence index: " + fileNum + " length: " + length + " path " + Dir);

            MotionDataControl(motionCaptureData);// set initial pose
            if (showResult == false)
            {
                RecordJointPostureToController();
                ImuData2logPosition();// record data
            }
            }
        
        if (i == length - 1 && adjustFrame == physicsLimit)// finished one motion cycle; continue or save
        {
            if (showResult == true)// loop playback for demo
            {
                if (Mathf.Abs(hips.position.x - m_JdController.bodyPartsDict[hips].startingPos.x) < 10)// target distance: 10m
                {
                    midStartPosition = new Vector3(hips.position.x - m_JdController.bodyPartsDict[hips].startingPos.x, 0, 0);
                }
                else
                {
                    fileNum++;
                    midStartPosition = Vector3.zero;
                    string Dir2 = Path.Combine(Application.streamingAssetsPath,"Rigidbody") + Dir.Remove(0, 67);
                    File.WriteAllText(Dir2, JsonConvert.SerializeObject(rbVelocity), new System.Text.UTF8Encoding(false));
                }
            }
            if (showResult == false)// when refining physics, save back to source
            {
                fileNum++;
                midStartPosition = Vector3.zero;

                UpdateIniVelocity();// fill first-frame velocity, update new hip height
                UpdateStepInformation();// use original step info directly
                FinalProcessing();
                Debug.Log("Average walking speed: " + imuData2.aveVelocity);

                // Save
                string Dir2 = Path.Combine(Application.streamingAssetsPath,"JsonWithP") + Dir.Remove(0, 62);
                File.WriteAllText(Dir2, JsonConvert.SerializeObject(imuData2), new System.Text.UTF8Encoding(false));
            }
            i = 0; adjustFrame = -1;

        }
        else if (adjustFrame == physicsLimit)// after each physics step, record hip height and move to next frame
        {
            if (i > 0 && i < length)// draw helper lines
            {
                Transform lineBody = footL;
                Debug.DrawLine(m_JdController.bodyPartsDict[lineBody].PrePosition, lineBody.position, Color.grey, 2);
                lineBody = footR;
                Debug.DrawLine(m_JdController.bodyPartsDict[lineBody].PrePosition, lineBody.position, Color.blue, 2);
            }

            if (i > 0 && (length - i) < 1 && showResult == false) TrailOnOff(false); else TrailOnOff(true);
            

            i++;

            MotionDataControl(motionCaptureData); // pose actor

            // Add distance-to-ground detection here
            if (showResult == false)
            {
                RecordJointVelocityToController();// log linear/angular velocity vs previous frame
                ImuData2logVolocity();// store angularVelocity/bodyPartVelocity in new dataset
                RecordJointPostureToController();// update controller pre-rotation/position (world space)
                ImuData2logPosition();// store data including hip drop and foot contact
            }

            adjustFrame = -1;
        }
        adjustFrame += 1;

    }

    private float Hipmove()
    {
        Vector3 footLv1 = footL.localToWorldMatrix.MultiplyPoint(new Vector3(0, 0.38f / 2 - 0.075f * 1.2f, 0) + footL.GetComponent<CapsuleCollider>().center);
        Vector3 footLv2 = footL.localToWorldMatrix.MultiplyPoint(new Vector3(0, -0.38f / 2 + 0.075f * 1.2f, 0) + footL.GetComponent<CapsuleCollider>().center);
        Vector3 footRv1 = footR.localToWorldMatrix.MultiplyPoint(new Vector3(0, 0.38f / 2 - 0.075f * 1.2f, 0) + footR.GetComponent<CapsuleCollider>().center);
        Vector3 footRv2 = footR.localToWorldMatrix.MultiplyPoint(new Vector3(0, -0.38f / 2 + 0.075f * 1.2f, 0) + footR.GetComponent<CapsuleCollider>().center);

        float r = Vector3.Magnitude(footL.localToWorldMatrix.MultiplyPoint(new Vector3(0, 0.075f * 1.2f, 0)) - footL.localToWorldMatrix.MultiplyPoint(new Vector3(0, 0, 0)));
        //Debug.DrawLine(footRv1, footRv2, Color.red, 1); Debug.DrawLine(footRv1, footRv2 + new Vector3(0, r, 0), Color.red, 1); Debug.DrawLine(footRv1 + new Vector3(0, r, 0), footRv2, Color.red, 1);

        float minheight = Mathf.Min(footLv1.y, footLv2.y, footRv1.y, footRv2.y);
        Vector3 radStart = new Vector3(footL.position.x, minheight - r, footL.position.z);
        Vector3 radStart2 = new Vector3(footR.position.x, minheight - r, footR.position.z);

        RaycastHit[] hit;// raycast hits

        List<float> rayDistancelist = new List<float>(); ;

        hit = Physics.RaycastAll(radStart, Vector3.down, 500f, ~(1 << 0));// with collisions


        for (int c = 0; c < hit.Length; c++)
        {
            if (hit[c].collider.gameObject.CompareTag("ground"))// collider is ground
            { rayDistancelist.Add(hit[c].distance); }
        }

        if (showResult == false)
        {
            Debug.DrawLine(radStart, radStart - new Vector3(0, rayDistancelist.Min(), 0), Color.blue, 0.01f);
            Debug.DrawLine(radStart2, radStart2 - new Vector3(0, rayDistancelist.Min(), 0), Color.blue, 0.01f);
        }

        imuData2.footContact[i, 0] = Mathf.Abs(footLv1.y - minheight) < 0.02f || Mathf.Abs(footLv2.y - minheight) < 0.02f ? 1 : 0;
        imuData2.footContact[i, 1] = Mathf.Abs(footRv1.y - minheight) < 0.02f || Mathf.Abs(footRv2.y - minheight) < 0.02f ? 1 : 0;
        return rayDistancelist.Min();// nearest ground distance
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
    /// Compute center of mass
    /// </summary>
    /// <returns>COM in world space</returns>
    private Vector3 ComCalculate()
    {
        Vector3 com = Vector3.zero; float mass = 0;
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {
             com += m_JdController.bodyPartsDict[trans].rb.mass * trans.position;
             mass += m_JdController.bodyPartsDict[trans].rb.mass;

        }
        com /= mass;// COM position
        return com;
    }

    /// <summary>
    /// For all objects tagged "mesh", check ground contact and swap material on contact
    /// </summary>
    public void GroundTouchVisualize()
    {

        foreach (var meshObject in meshChangeObject)
        {
            // find parent transform
            Transform parentTransform = meshObject.transform.parent;
            // if parent transform has a joint controller entry
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
    /// Store position data for the new dataset: walk (pose subset), footContact, endPosition, comPosition
    /// </summary>
    private void ImuData2logPosition()
    {
        // comPosition
        Vector3 com = ComCalculate()-hips.position;
        imuData2.comPosition[i, 0] = com.x; imuData2.comPosition[i, 1] = com.y; imuData2.comPosition[i, 2] = com.z;

        // endPosition
        List<Transform> bodyEnd = new List<Transform> { footL, footR, spine };
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
        imuData2.walk[i, 1] = hips.position.y - m_JdController.bodyPartsDict[hips].startingPos.y-Hipmove(); 
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
    /// <param name="preRoation">previous frame</param>
    /// <param name="thisRotation">current frame</param>
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
    /// Correct rotation order. Based on mocap data, get quaternion rotating about the root in XYZ order
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
    /// Set hip position and rotations for 8 joints from mocap dataset for frame i
    /// </summary>
    private void MotionDataControl(float[,] motionData )
    {
        if (showResult == false) MotionDataControlData1(motionData);// use raw angle data
        if (showResult == true) MotionDataControlData2(motionData);// use physics-adjusted quaternion data
        hips.position += midStartPosition;

        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.rb.velocity = Vector3.zero;
            bodyPart.rb.angularVelocity = Vector3.zero;
        }
    }

    private void MotionDataControlData1(float[,] motionData)// total 62 dims
    {
        // hip position is relative to initial pose
        hips.position = m_JdController.bodyPartsDict[hips].startingPos + Quaternion.AngleAxis(angleTowards, Vector3.up) * new Vector3(motionData[i, 0], motionData[i, 1], motionData[i, 2]);
        hips.rotation = m_JdController.bodyPartsDict[hips].startingRot * AngleRoatation(motionData[i, 3], -motionData[i, 4], -motionData[i, 5]);

        if (motionCaptureData[length - 1, 2] < motionCaptureData[0, 2]) hips.rotation*=Quaternion.AngleAxis(180, Vector3.up);

        thighR.localRotation = AngleRoatation(0, 0, 20) * AngleRoatation(motionData[i, 48], -motionData[i, 49], -motionData[i, 50])*AngleRoatation(0, 0, 3*0);
        shinR.localRotation = AngleRoatation(motionData[i, 51], 0, 0);
        thighL.localRotation = AngleRoatation(0, 0, -20) * AngleRoatation(motionData[i, 55], -motionData[i, 56], -motionData[i, 57])*AngleRoatation(0, 0, -3*0);
        shinL.localRotation = AngleRoatation(motionData[i, 58], 0, 0);
        spine.localRotation = AngleRoatation(-10, 0, 0) * AngleRoatation(motionData[i, 6], -motionData[i, 7], -motionData[i, 8]) * AngleRoatation(motionData[i, 9], -motionData[i, 10], -motionData[i, 11]);
        footR.localRotation = m_JdController.bodyPartsDict[footR].startingLocalRot * AngleRoatation(motionData[i, 52], 0, -motionData[i, 53]);
        footL.localRotation = m_JdController.bodyPartsDict[footL].startingLocalRot * AngleRoatation(motionData[i, 59], 0, -motionData[i, 60]);
    }

    private void MotionDataControlData2(float[,] motionData)// total 35 dims
    {
        hips.position = m_JdController.bodyPartsDict[hips].startingPos + new Vector3(motionData[i, 0], motionData[i, 1], motionData[i, 2]);

        int count = 3;
        foreach (var trans in bodylistInUse)//q order: x y z w
        {
            Quaternion bodyAngle = new Quaternion(motionData[i, count++], motionData[i, count++], motionData[i, count++], motionData[i, count++]);
            if (trans == hips) trans.rotation = bodyAngle;
            else trans.localRotation = bodyAngle;

            // position the debug cube
            if (trans == hips)
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
            if (filename.EndsWith(suffix))// check suffix and add matching file path
            {
                fileList.Add(filename);
            }
        }
        return fileList;
    }


    /// <summary>
    /// Record current full-body pose to JdController: rotation, position (world space), and foot contact
    /// </summary>
    public void RecordJointPostureToController()
    {
        float height = Hipmove();
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {

            // update position (world)
            m_JdController.bodyPartsDict[trans].PrePosition = trans.position - new Vector3(0, height, 0);

            // update rotation (world)
            m_JdController.bodyPartsDict[trans].PreRoation = trans.rotation;
         
        }

        m_JdController.bodyPartsDict[footL].preContact = m_JdController.bodyPartsDict[footL].groundContact.touchingGround;
        m_JdController.bodyPartsDict[footR].preContact = m_JdController.bodyPartsDict[footR].groundContact.touchingGround;

    }

    /// <summary>
    /// Update step length/frequency; now read directly from MATLAB output instead of computing here
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
    /// Update linear/angular velocity in bodyPart using current pose and prior RecordPosture()
    /// </summary>
    private void RecordJointVelocityToController()
    {
        float height = Hipmove();
        foreach (var trans in m_JdController.bodyPartsDict.Keys)
        {
           // linear velocity
           m_JdController.bodyPartsDict[trans].velocity = (trans.position + new Vector3(0, -height, 0) - m_JdController.bodyPartsDict[trans].PrePosition) / Time.fixedDeltaTime;

            // angular velocity
            m_JdController.bodyPartsDict[trans].angularVelocity = AngularVelocityCalculate(m_JdController.bodyPartsDict[trans].PreRoation, trans.rotation);
        }
    }

    /// <summary>
    /// Apply moving-average smoothing to final data
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
    /// Moving average for binary contact
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
    /// Moving average smoothing
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
