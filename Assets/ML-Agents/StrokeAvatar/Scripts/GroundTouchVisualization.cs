using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Attach to mesh object, changes color on ground contact
/// </summary>
public class GroundTouchVisualization : MonoBehaviour
{
    GameObject parentObject;

    // Start is called before the first frame update
    void Start()
    {
        parentObject = transform.parent.gameObject;
        Debug.Log(parentObject.name);
    }

    // Update is called once per frame
    void Update()
    {

    }
}
