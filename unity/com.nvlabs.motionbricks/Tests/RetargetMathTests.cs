using MotionBricks;
using NUnit.Framework;
using UnityEngine;

namespace MotionBricks.Tests
{
    /// <summary>
    /// Regression tests for the retargeter's support math: the reference IK easing ramp and the
    /// scene/SOMA space map the retargeter exposes to samples.
    /// </summary>
    public sealed class RetargetMathTests
    {
        /// <summary>
        /// The IK ease stays zero for the first half of a segment and reaches 1 at the transition:
        /// the reference character uses smoothstep(0.5, 1, progress).
        /// </summary>
        [Test]
        public void LateRampWeightMatchesReference()
        {
            var progress = new[] { 0f, 0.1f, 0.25f, 0.5f, 0.6f, 0.75f, 0.9f, 1f };
            var expected = new[] { 0f, 0f, 0f, 0f, 0.104f, 0.5f, 0.896f, 1f };
            for (var index = 0; index < progress.Length; index++)
            {
                Assert.AreEqual(
                    expected[index], TargetPoseIkCorrector.LateRampWeight(progress[index]), 1e-3,
                    $"Progress {progress[index]} eased to the wrong weight.");
            }

            // Progress clamps outside [0, 1].
            Assert.AreEqual(0f, TargetPoseIkCorrector.LateRampWeight(-0.5f), 1e-6);
            Assert.AreEqual(1f, TargetPoseIkCorrector.LateRampWeight(1.5f), 1e-6);
        }

        /// <summary>The retargeter's scene map maps exactly the anchored, scaled segment.</summary>
        [Test]
        public void SomaSceneMapMatchesAnchorAndScale()
        {
            var map = new SomaSceneMap(new Vector3(1.5f, 0.9f, -2f), new Vector3(4f, 1.05f, 6f), 0.96f);

            // The anchor maps onto itself.
            Assert.AreEqual(new Vector3(1.5f, 0.9f, -2f), map.SomaPointToScene(new Vector3(4f, 1.05f, 6f)));

            // A displaced point scales by the effective scale.
            var scene = map.SomaPointToScene(new Vector3(4f, 2.05f, 6.5f));
            Assert.AreEqual(1.5f, scene.x, 1e-5);
            Assert.AreEqual(0.9f + 1f * 0.96f, scene.y, 1e-5);
            Assert.AreEqual(-2f + 0.5f * 0.96f, scene.z, 1e-5);

            // Round trips hold for points mapped scene-ward and back.
            for (var offset = 1; offset < 6; offset++)
            {
                var somaPoint = new Vector3(4f + offset, 1.05f + offset * 0.3f, 6f - offset * 0.7f);
                Assert.IsTrue(
                    Distance(map.ScenePointToSoma(map.SomaPointToScene(somaPoint)), somaPoint) < 1e-4,
                    $"SOMA point {somaPoint} did not round trip.");

                var scenePoint = new Vector3(-2f + offset * 0.5f, 0.1f * offset, 3f - offset);
                Assert.IsTrue(
                    Distance(map.SomaPointToScene(map.ScenePointToSoma(scenePoint)), scenePoint) < 1e-4,
                    $"Scene point {scenePoint} did not round trip.");
            }

            // Distances scale by the map's scale.
            var a = map.SomaPointToScene(Vector3.zero);
            var b = map.SomaPointToScene(Vector3.right);
            Assert.AreEqual(0.96f, Vector3.Distance(a, b), 1e-4);
        }

        /// <summary>
        /// With the retargeter's height scale, the SOMA ground maps onto the avatar's ground: the
        /// map builds from hips heights above the avatar root.
        /// </summary>
        [Test]
        public void SomaSceneMapKeepsGroundPlane()
        {
            // An avatar whose hips sit 0.96 m above its root, seeded against a SOMA hips at 1.012 m.
            var hipsHeight = 0.96f;
            var somaHipsHeight = 1.012f;
            var scale = hipsHeight / somaHipsHeight;
            var map = new SomaSceneMap(new Vector3(0f, hipsHeight, 0f), new Vector3(0f, somaHipsHeight, 0f), scale);

            var sceneGround = map.SomaPointToScene(new Vector3(0.3f, 0f, -0.4f));
            Assert.AreEqual(0f, sceneGround.y, 1e-4, "The SOMA ground must map onto the avatar's ground.");

            var sceneHips = map.SomaPointToScene(new Vector3(0f, somaHipsHeight, 0f));
            Assert.AreEqual(hipsHeight, sceneHips.y, 1e-4);
        }

        /// <summary>Directions are unchanged: the map is a translation and a uniform scale only.</summary>
        [Test]
        public void SomaSceneMapKeepsDirections()
        {
            var map = new SomaSceneMap(new Vector3(-3f, 0.2f, 12f), new Vector3(1f, 1f, 1f), 0.8f);
            var directions = new[]
            {
                Vector3.forward, Vector3.right, Vector3.up, new Vector3(1f, 0f, -1f).normalized,
            };

            foreach (var direction in directions)
            {
                var mapped = (map.SomaPointToScene(direction) - map.SomaPointToScene(Vector3.zero)).normalized;
                Assert.AreEqual(1f, Vector3.Dot(direction, mapped), 1e-5, $"Direction {direction} changed.");
            }
        }

        /// <summary>The identity map equals the identity transform.</summary>
        [Test]
        public void SomaSceneMapIdentityLeavesPointsAlone()
        {
            var map = SomaSceneMap.Identity;
            var point = new Vector3(1.3f, -0.7f, 2.2f);
            Assert.AreEqual(point, map.SomaPointToScene(point));
            Assert.AreEqual(point, map.ScenePointToSoma(point));
        }

        [TestCase(0f)]
        [TestCase(90f)]
        [TestCase(180f)]
        public void SomaSceneMapRotatesAndRoundTrips(float yaw)
        {
            var rotation = Quaternion.Euler(0f, yaw, 0f);
            var map = new SomaSceneMap(new Vector3(3f, 5f, -2f), new Vector3(-1f, 1f, 2f), 0.8f, rotation);
            var source = new Vector3(2f, 0.5f, -4f);
            var scene = map.SomaPointToScene(source);
            Assert.That(Vector3.Distance(map.ScenePointToSoma(scene), source), Is.LessThan(1e-5f));
            Assert.That(Vector3.Distance(map.SomaDirectionToScene(Vector3.forward), rotation * Vector3.forward), Is.LessThan(1e-5f));
            Assert.That(Vector3.Distance(map.SceneDirectionToSoma(rotation * Vector3.right), Vector3.right), Is.LessThan(1e-5f));
        }

        [Test]
        public void JointScaleIsIndependentOfRootTravelScale()
        {
            var map = new SomaSceneMap(Vector3.up, Vector3.up, 2f, Quaternion.Euler(0f, 90f, 0f));
            var root = new Vector3(0f, 1f, 1f);
            var joint = root + Vector3.forward;
            var rootScene = map.SomaPointToScene(root);
            var jointScene = map.SomaJointToScene(root, joint, 0.8f);
            Assert.That(Vector3.Distance(rootScene, jointScene), Is.EqualTo(0.8f).Within(1e-5f));
            Assert.That(Vector3.Distance(map.ScenePointToSomaJoint(root, jointScene, 0.8f), joint), Is.LessThan(1e-5f));
        }

        [Test]
        public void SomaSceneMapRejectsInvalidScales()
        {
            Assert.Throws<System.ArgumentOutOfRangeException>(() => new SomaSceneMap(Vector3.zero, Vector3.zero, 0f));
            Assert.Throws<System.ArgumentOutOfRangeException>(() => new SomaSceneMap(Vector3.zero, Vector3.zero, float.NaN));
            Assert.Throws<System.ArgumentOutOfRangeException>(() => SomaSceneMap.Identity.ScenePointToSomaJoint(Vector3.zero, Vector3.one, 0f));
        }

        private static float Distance(Vector3 a, Vector3 b)
        {
            return (a - b).magnitude;
        }
    }
}
