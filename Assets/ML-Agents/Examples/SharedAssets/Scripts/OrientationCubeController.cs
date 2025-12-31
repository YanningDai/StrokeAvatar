using UnityEngine;

namespace Unity.MLAgentsExamples
{
    /// <summary>
    /// Utility class to allow a stable observation platform.
    /// </summary>
    public class OrientationCubeController : MonoBehaviour
    {
        //Update position and Rotation
        public Transform hips;
        private Quaternion startRotation;
        private Vector3 startPosition;
        private Vector3 startHipPosition;
        public void Awake()
        {
            startRotation = transform.rotation;
            startPosition = transform.position;
            startHipPosition = hips.position;
        }

        public void FixedUpdate()
        {
            transform.position = hips.position + startPosition - startHipPosition;
            transform.rotation = startRotation;
        }
        public void UpdateOrientation(Transform rootBP, Transform target)//其他函数用的
        {
            var dirVector = target.position - transform.position;
            dirVector.y = 0; //flatten dir on the y. this will only work on level, uneven surfaces
            var lookRot =
                dirVector == Vector3.zero
                    ? Quaternion.identity
                    : Quaternion.LookRotation(dirVector); //get our look rot to the target
            //UPDATE ORIENTATION CUBE POS & ROT
            transform.SetPositionAndRotation(rootBP.position, lookRot);
        }
    }
}
