namespace MotionBricks
{
    /// <summary>
    /// Timing and shape constants that the MotionBricks SOMA model shares, ported from
    /// <c>motionbricks/model/contract.py</c>.
    /// </summary>
    public static class MbContract
    {
        /// <summary>Sample rate of the model's poses, in Hz.</summary>
        public const float SampleRate = 30f;

        /// <summary>Poses each token holds.</summary>
        public const int PosesPerToken = 4;

        /// <summary>Context plus target poses submit per generation (4 + 4).</summary>
        public const int ContextAndTargetPoses = 2 * PosesPerToken;

        /// <summary>Fewest motion tokens a generation may produce.</summary>
        public const int MinTokenCount = 6;

        /// <summary>Most motion tokens a generation may produce.</summary>
        public const int MaxTokenCount = 16;

        /// <summary>Frames removed from each end of the generated motion.</summary>
        public const int BoundaryFrames = PosesPerToken - 1;

        /// <summary>Foot contact values per frame (left heel, left toe, right heel, right toe).</summary>
        public const int FootContactCount = 4;

        /// <summary>The default allowed token-count range of <see cref="MotionBricksInferenceConfig"/>.</summary>
        public const int DefaultMinTokenCount = 8;
        public const int DefaultMaxTokenCount = 14;

        /// <summary>Most valid frames a generation yields (with the maximum token count).</summary>
        public const int MaxValidFrames = MaxTokenCount * PosesPerToken - 2 * BoundaryFrames;

        /// <summary>Returns the valid frame count left after removing the boundary frames.</summary>
        public static int ValidFrameCount(int tokenCount)
        {
            if (tokenCount < MinTokenCount || tokenCount > MaxTokenCount)
            {
                throw new System.ArgumentOutOfRangeException(
                    nameof(tokenCount), tokenCount, $"tokenCount must be {MinTokenCount} to {MaxTokenCount}.");
            }

            return tokenCount * PosesPerToken - 2 * BoundaryFrames;
        }

        /// <summary>Returns the allowed token counts as a bit mask, with bit 0 for 6 tokens.</summary>
        public static int AllowedTokenBits(System.Collections.Generic.IReadOnlyCollection<int> counts)
        {
            if (counts == null || counts.Count == 0)
            {
                throw new System.ArgumentException("allowedTokenCounts cannot be empty.", nameof(counts));
            }

            var bits = 0;
            foreach (var value in counts)
            {
                if (value < MinTokenCount || value > MaxTokenCount)
                {
                    throw new System.ArgumentException(
                        $"allowedTokenCounts values must be {MinTokenCount} to {MaxTokenCount}, got {value}.");
                }

                bits |= 1 << (value - MinTokenCount);
            }

            return bits;
        }
    }
}
