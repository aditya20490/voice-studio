package com.voicestudio.app

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/** Records mono 16-bit WAV at 24 kHz (the rate voice-cloning models expect). */
class WavRecorder {
    companion object {
        const val SAMPLE_RATE = 24000
    }

    @Volatile
    var isRecording = false
        private set

    @Volatile
    var seconds = 0
        private set

    @Volatile
    var level = 0f
        private set

    private var worker: Thread? = null

    @SuppressLint("MissingPermission")
    fun start(out: File) {
        if (isRecording) return
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val rec = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, SAMPLE_RATE)
        )
        isRecording = true
        seconds = 0
        worker = Thread {
            val raf = RandomAccessFile(out, "rw")
            raf.setLength(0)
            raf.write(ByteArray(44)) // header placeholder
            val buf = ByteArray(4096)
            var total = 0L
            rec.startRecording()
            while (isRecording) {
                val n = rec.read(buf, 0, buf.size)
                if (n > 0) {
                    raf.write(buf, 0, n)
                    total += n
                    seconds = (total / (SAMPLE_RATE * 2)).toInt()
                    var peak = 0
                    var i = 0
                    while (i + 1 < n) {
                        val s = (buf[i].toInt() and 0xFF) or (buf[i + 1].toInt() shl 8)
                        peak = maxOf(peak, abs(s.toShort().toInt()))
                        i += 2
                    }
                    level = peak / 32768f
                }
            }
            rec.stop()
            rec.release()
            raf.seek(0)
            raf.write(header(total))
            raf.close()
        }.also { it.start() }
    }

    fun stop() {
        isRecording = false
        worker?.join()
        worker = null
        level = 0f
    }

    private fun header(dataLen: Long): ByteArray {
        val b = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray())
        b.putInt((36 + dataLen).toInt())
        b.put("WAVE".toByteArray())
        b.put("fmt ".toByteArray())
        b.putInt(16)
        b.putShort(1) // PCM
        b.putShort(1) // mono
        b.putInt(SAMPLE_RATE)
        b.putInt(SAMPLE_RATE * 2)
        b.putShort(2)
        b.putShort(16)
        b.put("data".toByteArray())
        b.putInt(dataLen.toInt())
        return b.array()
    }
}
