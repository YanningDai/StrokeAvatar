using UnityEngine;

public class SlopeController : MonoBehaviour
{
    [Header("Slope Geometry (drag references)")]
    [SerializeField] private Transform flat;
    [SerializeField] private Transform up;

    [Header("Slope Parameters")]
    public float rampLength = 9f; // meters

    [Header("Curriculum (read-only)")]
    [SerializeField, Tooltip("Current slope height (meters)")]
    private float currentHeight;

    public void ApplyCurriculum(float height)
    {
        if (flat == null || up == null)
            return;

        currentHeight = height; 
        // 1. flat height
        Vector3 flatPos = flat.localPosition;
        flatPos.y = height;
        flat.localPosition = flatPos;

        // 2. slope angle
        float angleDeg = Mathf.Atan(height / rampLength) * Mathf.Rad2Deg;
        up.localRotation = Quaternion.Euler(0f, 0f, angleDeg);

        // 3. slope vertical alignment
        Collider upCol = up.GetComponent<Collider>();
        Collider flatCol = flat.GetComponent<Collider>();
        if (upCol == null || flatCol == null) return;

        Bounds flatBounds = flatCol.bounds;
        Vector3 flatTop = new Vector3(
            flatBounds.min.x,
            flatBounds.max.y,
            flatBounds.center.z
        );

        BoxCollider upBox = upCol as BoxCollider;
        if (upBox == null) return;

        Vector3 upTopLocal = new Vector3(
            upBox.center.x + upBox.size.x * 0.5f,
            upBox.center.y + upBox.size.y * 0.5f,
            upBox.center.z
        );

        Vector3 upTop = up.TransformPoint(upTopLocal);
        up.position += flatTop - upTop;

    }
}
