using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace MotionBricks.Tests
{
    /// <summary>
    /// A minimal JSON reader for the test fixtures. The fixture format uses nested arrays and objects,
    /// which Unity's JsonUtility cannot read, so the tests parse it with this reader.
    /// </summary>
    internal sealed class MiniJson
    {
        private readonly string _text;
        private int _index;

        private MiniJson(string text)
        {
            _text = text;
        }

        public static MiniJson Parse(string text)
        {
            var json = new MiniJson(text);
            json.SkipWhitespace();
            return json;
        }

        /// <summary>Reads the value at the current position.</summary>
        public object ReadValue()
        {
            SkipWhitespace();
            var c = _text[_index];
            switch (c)
            {
                case '{':
                    return ReadObject();
                case '[':
                    return ReadArray();
                case '"':
                    return ReadString();
                case 't':
                    Expect("true");
                    return true;
                case 'f':
                    Expect("false");
                    return false;
                case 'n':
                    Expect("null");
                    return null;
                default:
                    return ReadNumber();
            }
        }

        private Dictionary<string, object> ReadObject()
        {
            var result = new Dictionary<string, object>(StringComparer.Ordinal);
            _index++; // '{'
            SkipWhitespace();
            if (_text[_index] == '}')
            {
                _index++;
                return result;
            }

            while (true)
            {
                SkipWhitespace();
                var key = ReadString();
                SkipWhitespace();
                _index++; // ':'
                var value = ReadValue();
                result[key] = value;
                SkipWhitespace();
                var c = _text[_index++];
                if (c == '}')
                {
                    return result;
                }
            }
        }

        private List<object> ReadArray()
        {
            var result = new List<object>();
            _index++; // '['
            SkipWhitespace();
            if (_text[_index] == ']')
            {
                _index++;
                return result;
            }

            while (true)
            {
                result.Add(ReadValue());
                SkipWhitespace();
                var c = _text[_index++];
                if (c == ']')
                {
                    return result;
                }
            }
        }

        private string ReadString()
        {
            _index++; // '"'
            var builder = new StringBuilder();
            while (true)
            {
                var c = _text[_index++];
                if (c == '"')
                {
                    return builder.ToString();
                }

                if (c == '\\')
                {
                    var escaped = _text[_index++];
                    builder.Append(escaped switch
                    {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        'b' => '\b',
                        'f' => '\f',
                        '"' => '"',
                        '\\' => '\\',
                        '/' => '/',
                        _ => escaped,
                    });
                    continue;
                }

                builder.Append(c);
            }
        }

        private double ReadNumber()
        {
            var start = _index;
            while (_index < _text.Length && "+-0123456789.eE".IndexOf(_text[_index]) >= 0)
            {
                _index++;
            }

            return double.Parse(_text.Substring(start, _index - start), CultureInfo.InvariantCulture);
        }

        private void Expect(string literal)
        {
            if (string.CompareOrdinal(_text, _index, literal, 0, literal.Length) != 0)
            {
                throw new FormatException($"Expected '{literal}' at {_index}.");
            }

            _index += literal.Length;
        }

        private void SkipWhitespace()
        {
            while (_index < _text.Length && char.IsWhiteSpace(_text[_index]))
            {
                _index++;
            }
        }
    }

    /// <summary>Helpers that read the fixture's JSON values.</summary>
    internal static class MiniJsonExtensions
    {
        public static Dictionary<string, object> AsObject(this object value)
        {
            return (Dictionary<string, object>)value;
        }

        public static List<object> AsArray(this object value)
        {
            return (List<object>)value;
        }

        public static double AsNumber(this object value)
        {
            return (double)value;
        }

        public static string AsString(this object value)
        {
            return (string)value;
        }

        public static float[] AsFloatArray(this object value)
        {
            var array = value.AsArray();
            var result = new float[array.Count];
            for (var index = 0; index < array.Count; index++)
            {
                result[index] = (float)array[index].AsNumber();
            }

            return result;
        }

        /// <summary>Reads a list of number lists as one flat array.</summary>
        public static float[] AsFlatFloatArray(this object value)
        {
            var rows = value.AsArray();
            var result = new List<float>();
            foreach (var row in rows)
            {
                if (row is List<object> rowValues)
                {
                    foreach (var entry in rowValues)
                    {
                        result.Add((float)entry.AsNumber());
                    }
                }
                else
                {
                    result.Add((float)row.AsNumber());
                }
            }

            return result.ToArray();
        }

        public static int[] AsIntArray(this object value)
        {
            var array = value.AsArray();
            var result = new int[array.Count];
            for (var index = 0; index < array.Count; index++)
            {
                result[index] = (int)Math.Round(array[index].AsNumber());
            }

            return result;
        }

        /// <summary>Reads a pose object as a position and a rotation.</summary>
        public static UnityEngine.Pose AsPose(this object value)
        {
            var pose = value.AsObject();
            return new UnityEngine.Pose(
                new UnityEngine.Vector3(
                    (float)pose["px"].AsNumber(),
                    (float)pose["py"].AsNumber(),
                    (float)pose["pz"].AsNumber()),
                new UnityEngine.Quaternion(
                    (float)pose["qx"].AsNumber(),
                    (float)pose["qy"].AsNumber(),
                    (float)pose["qz"].AsNumber(),
                    (float)pose["qw"].AsNumber()));
        }
    }
}
