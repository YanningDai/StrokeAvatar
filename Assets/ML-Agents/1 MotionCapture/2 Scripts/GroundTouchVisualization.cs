using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 挂载在mesh文件上，触地就改变颜色
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
