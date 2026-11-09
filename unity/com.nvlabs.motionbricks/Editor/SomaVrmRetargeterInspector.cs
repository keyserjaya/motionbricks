using System.Text;
using MotionBricks;
using UnityEditor;
using UnityEngine;

namespace MotionBricks.Editor
{
    /// <summary>
    /// Inspector for the SOMA VRM retargeter: the driving knobs, plus live diagnostics of the
    /// calibration, the scene map, and the avatar limb IK pins, so retargeting can be debugged
    /// without recompiling.
    /// </summary>
    [CustomEditor(typeof(SomaVrmRetargeter))]
    public sealed class SomaVrmRetargeterInspector : UnityEditor.Editor
    {
        private bool _showDiagnostics = true;

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            var retargeter = (SomaVrmRetargeter)target;
            _showDiagnostics = EditorGUILayout.Foldout(_showDiagnostics, "Live Diagnostics", true);
            if (!_showDiagnostics)
            {
                return;
            }

            using (new EditorGUI.DisabledScope(true))
            {
                EditorGUILayout.TextField("Initialized", retargeter.IsInitialized ? "Yes" : "No (not in rest pose / invalid avatar)");
                EditorGUILayout.FloatField("Height Scale", retargeter.HeightScale);
                EditorGUILayout.FloatField("Root Scale (motion)", retargeter.RootScale);
                EditorGUILayout.FloatField("Effective Scene Scale", retargeter.EffectiveRootScale);
                EditorGUILayout.FloatField("Placement Yaw (deg)", retargeter.PlacementRotation.eulerAngles.y);
                EditorGUILayout.Toggle("Swap Left/Right Binding", retargeter.SwapLeftRightBones);
                EditorGUILayout.Vector3Field("Anchor (scene)", retargeter.AnchorPosition);
                EditorGUILayout.Vector3Field("Anchor (SOMA)", retargeter.AnchorSomaPosition);
                EditorGUILayout.IntField("Bound Bones", retargeter.BoundBoneCount);
                EditorGUILayout.FloatField("Avatar Limb Pin Error (m)", retargeter.AvatarLimbPinError);
                EditorGUILayout.TextField("Avatar Limb IK", retargeter.AvatarLimbPinActive
                    ? "Solving (pins active)"
                    : "Idle (no pins or weight 0)");
                EditorGUILayout.TextField("Missing Bones", string.IsNullOrEmpty(retargeter.MissingBonesReport)
                    ? "None — all humanoid bones bound"
                    : retargeter.MissingBonesReport);
            }

            var map = retargeter.CurrentSceneMap;
            EditorGUILayout.HelpBox(
                $"Scene = anchor + yaw * (SOMA - anchorSoma) * {map.Scale:F3}\n" +
                $"Placement yaw: {map.Rotation.eulerAngles.y:F1} degrees. Source limb gizmos use height scale, not root scale." +
                DiagnosticsHint(retargeter), MessageType.Info, true);

            if (Application.isPlaying)
            {
                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button("Rebase Current Poses"))
                {
                    retargeter.Recalibrate();
                    EditorUtility.SetDirty(retargeter);
                }

                if (GUILayout.Button("Reinitialize (apply toggles)"))
                {
                    retargeter.Reinitialize();
                    EditorUtility.SetDirty(retargeter);
                }

                EditorGUILayout.EndHorizontal();
            }
        }

        private static string DiagnosticsHint(SomaVrmRetargeter retargeter)
        {
            var builder = new StringBuilder();
            if (!retargeter.IsInitialized)
            {
                builder.Append("\n\nInitialize in Play mode with the avatar in its rest pose (no controller playing).");
            }
            else if (Mathf.Abs(retargeter.HeightScale - 1f) > 0.4f)
            {
                builder.Append(
                    $"\n\nThe avatar is {(retargeter.HeightScale > 1f ? "taller" : "shorter")} than the SOMA " +
                    "skeleton, so root motion scales accordingly. Unusually small or large scales point at a " +
                    "bad rest pose or a warp-scale avatar.");
            }

            return builder.ToString();
        }
    }
}
