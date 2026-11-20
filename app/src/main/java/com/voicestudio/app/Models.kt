package com.voicestudio.app

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/** One-time download of the model files. After this, the app never touches the internet. */
object Models {
    private const val Q4 = "https://huggingface.co/Vocabook/chatterbox-multilingual-ONNX-q4/resolve/main/onnx/"
    private const val TOK = "https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX/resolve/main/tokenizer.json"

    private val files = listOf(
        "tokenizer.json" to TOK,
        "speech_encoder.onnx" to Q4 + "speech_encoder.onnx",
        "embed_tokens.onnx" to Q4 + "embed_tokens.onnx",
        "language_model.onnx" to Q4 + "language_model.onnx",
        "conditional_decoder.onnx" to Q4 + "conditional_decoder.onnx"
    )

    var downloading by mutableStateOf(false)
        private set
    var dlStatus by mutableStateOf("")
        private set

    fun dir(ctx: Context): File = File(ctx.filesDir, "models").apply { mkdirs() }

    fun file(ctx: Context, name: String): File = File(dir(ctx), name)

    fun isReady(ctx: Context): Boolean = files.all { File(dir(ctx), it.first).exists() }

    fun startDownload(ctx: Context) {
        if (downloading) return
        downloading = true
        dlStatus = "Starting..."
        val app = ctx.applicationContext
        Thread {
            try {
                download(app)
                dlStatus = "All models downloaded."
            } catch (e: Exception) {
                dlStatus = "Failed: ${e.message}. Tap Download again to resume."
            } finally {
                downloading = false
            }
        }.start()
    }

    private fun download(ctx: Context) {
        val dir = dir(ctx)
        for ((name, url) in files) {
            val out = File(dir, name)
            if (out.exists()) continue
            val part = File(dir, "$name.part")
            val have = if (part.exists()) part.length() else 0L
            val c = URL(url).openConnection() as HttpURLConnection
            c.connectTimeout = 20000
            c.readTimeout = 30000
            if (have > 0) c.setRequestProperty("Range", "bytes=$have-")
            c.connect()
            val code = c.responseCode
            if (code != 200 && code != 206) throw IOException("HTTP $code for $name")
            val resume = code == 206
            val base = if (resume) have else 0L
            val total = c.contentLengthLong + base
            var done = base
            var lastShown = 0L
            c.inputStream.use { input ->
                FileOutputStream(part, resume).use { o ->
                    val buf = ByteArray(64 * 1024)
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        o.write(buf, 0, n)
                        done += n
                        if (done - lastShown > 512 * 1024) {
                            lastShown = done
                            dlStatus = "Downloading $name: ${done / 1048576} / ${total / 1048576} MB"
                        }
                    }
                }
            }
            if (!part.renameTo(out)) throw IOException("Could not save $name")
        }
    }
}
