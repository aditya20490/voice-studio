package com.voicestudio.app

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Loads any audio file as mono 24 kHz floats. */
object AudioLoader {
    const val SR = 24000

    fun loadAll(clips: List<File>, maxSeconds: Int): FloatArray {
        val max = SR * maxSeconds
        var acc = FloatArray(0)
        for (c in clips) {
            acc += load(c)
            if (acc.size >= max) break
        }
        return acc.copyOf(minOf(acc.size, max))
    }

    private fun load(f: File): FloatArray {
        val (pcm, sr, ch) = tryWav(f) ?: decode(f)
        val frames = pcm.size / ch
        val mono = FloatArray(frames) { i ->
            var sum = 0f
            for (c in 0 until ch) sum += pcm[i * ch + c]
            sum / ch / 32768f
        }
        if (sr == SR || frames < 2) return mono
        val ratio = sr.toDouble() / SR
        val n = (frames / ratio).toInt()
        return FloatArray(n) { i ->
            val x = i * ratio
            val i0 = x.toInt()
            val i1 = minOf(i0 + 1, frames - 1)
            val fr = (x - i0).toFloat()
            mono[i0] * (1 - fr) + mono[i1] * fr
        }
    }

    private fun tryWav(f: File): Triple<ShortArray, Int, Int>? {
        val b = f.readBytes()
        if (b.size < 44 || String(b, 0, 4, Charsets.US_ASCII) != "RIFF" ||
            String(b, 8, 4, Charsets.US_ASCII) != "WAVE"
        ) return null
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        var p = 12
        var ch = 1
        var sr = SR
        var bits = 16
        var tag = 1
        while (p + 8 <= b.size) {
            val id = String(b, p, 4, Charsets.US_ASCII)
            val size = bb.getInt(p + 4)
            if (id == "fmt ") {
                tag = bb.getShort(p + 8).toInt()
                ch = bb.getShort(p + 10).toInt()
                sr = bb.getInt(p + 12)
                bits = bb.getShort(p + 22).toInt()
            } else if (id == "data") {
                if (tag != 1 || bits != 16 || ch < 1) return null
                val len = minOf(size.toLong() and 0xFFFFFFFFL, (b.size - p - 8).toLong()).toInt()
                val sb = ByteBuffer.wrap(b, p + 8, len).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                val s = ShortArray(sb.remaining())
                sb.get(s)
                return Triple(s, sr, ch)
            }
            if (size < 0) return null
            p += 8 + size + (size and 1)
        }
        return null
    }

    private fun decode(f: File): Triple<ShortArray, Int, Int> {
        val ex = MediaExtractor()
        ex.setDataSource(f.absolutePath)
        var track = -1
        var fmt: MediaFormat? = null
        for (i in 0 until ex.trackCount) {
            val tf = ex.getTrackFormat(i)
            if (tf.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                track = i
                fmt = tf
                break
            }
        }
        require(track >= 0 && fmt != null) { "No audio found in ${f.name}" }
        ex.selectTrack(track)
        var sr = fmt.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var ch = fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val codec = MediaCodec.createDecoderByType(fmt.getString(MediaFormat.KEY_MIME)!!)
        codec.configure(fmt, null, null, 0)
        codec.start()
        val info = MediaCodec.BufferInfo()
        val out = ByteArrayOutputStream()
        var inDone = false
        var outDone = false
        while (!outDone) {
            if (!inDone) {
                val i = codec.dequeueInputBuffer(10000)
                if (i >= 0) {
                    val buf = codec.getInputBuffer(i)!!
                    val n = ex.readSampleData(buf, 0)
                    if (n < 0) {
                        codec.queueInputBuffer(i, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inDone = true
                    } else {
                        codec.queueInputBuffer(i, 0, n, ex.sampleTime, 0)
                        ex.advance()
                    }
                }
            }
            val o = codec.dequeueOutputBuffer(info, 10000)
            if (o >= 0) {
                val ob = codec.getOutputBuffer(o)!!
                val chunk = ByteArray(info.size)
                ob.position(info.offset)
                ob.get(chunk)
                out.write(chunk)
                codec.releaseOutputBuffer(o, false)
                if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outDone = true
            } else if (o == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                val nf = codec.outputFormat
                sr = nf.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                ch = nf.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            }
        }
        codec.stop()
        codec.release()
        ex.release()
        val sb = ByteBuffer.wrap(out.toByteArray()).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val s = ShortArray(sb.remaining())
        sb.get(s)
        return Triple(s, sr, ch)
    }
}
