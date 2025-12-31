using UnityEngine;
using Random = UnityEngine.Random;
using Unity.MLAgents;
using UnityEngine.Events;


    /// <summary>
    /// Utility class to allow target placement and collision detection with an agent
    /// Add this script to the target you want the agent to touch.
    /// Callbacks will be triggered any time the target is touched with a collider tagged as 'tagToDetect'
    /// </summary>
public class StaticTarget : MonoBehaviour
{
    [Header("Collider Tag To Detect")]
    public string tagToDetect = "agent"; //collider tag to detect 
    private Vector3 m_startingPos; //the starting position of the target

    // Start is called before the first frame update
    void OnEnable()
    {
        m_startingPos = transform.position;
    }
        
    private void OnCollisionEnter(Collision col)
    {
        if (col.transform.CompareTag(tagToDetect))
        {
            var newTargetPos = m_startingPos;
            transform.position = newTargetPos;
        }
    }
}

