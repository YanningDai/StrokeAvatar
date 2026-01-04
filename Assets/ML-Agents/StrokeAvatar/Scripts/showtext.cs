using System.Collections;
using System.Collections.Generic;
using static Imitation;
using UnityEngine;
using UnityEngine.UI;

public class showtext : MonoBehaviour
{
    public GameObject text;
    public float timeLimit;
    private float total_time;
    public bool flagActive;
    // Start is called before the first frame update
    void Start()
    {
        text.SetActive(false);
        total_time = 0;
        flagActive = false;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (ImitationTestFive.showtext == true)
        {
            flagActive = true;
        }
        if (flagActive == true)
        {
            text.SetActive(true);
            total_time += Time.deltaTime;
            if (total_time > timeLimit)
            {
                text.SetActive(false);
                flagActive = false;
                total_time = 0;
            }
        }

    }
}
