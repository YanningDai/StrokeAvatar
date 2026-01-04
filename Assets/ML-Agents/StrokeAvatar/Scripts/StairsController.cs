using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class StairsController : MonoBehaviour
{
    private List<Transform> steps;

    void Awake()
    {
        steps = new List<Transform>();

        foreach (Transform child in transform)
        {
            if (child.name.StartsWith("Step"))
                steps.Add(child);
        }

        steps = steps
            .OrderBy(t =>
            {
                string num = t.name.Replace("Step", "");
                int.TryParse(num, out int idx);
                return idx;
            })
            .ToList();
    }

    public void ApplyCurriculum(float stepHeight)
    {
        if (steps == null || steps.Count == 0)
            return;

        for (int i = 0; i < steps.Count; i++)
        {
            Vector3 pos = steps[i].localPosition;
            pos.y = (i + 1) * stepHeight/100;
            steps[i].localPosition = pos;
        }
    }
}

