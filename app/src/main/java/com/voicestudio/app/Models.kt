package com.voicestudio.app

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import org.json.JSONArray
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/** One-time download of the model files. After this, the app never touches the internet. */
object Models {
    private val REPOS = listOf(
        "Vocabook/chatterbox-multilingual-ONNX-q4",
        "ipsilondev/chatterbox-multilingual-ONNX-q4",
        "onnx-community/chatterbox-multilingual-ONNX"
    )
    private val BASES = listOf("speech_encoder", "embed_tokens", "language_model", "conditional_decoder")
    private const val TOK = "https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX/resolve/main/tokenizer.json"

    private class Item(val repo: String, val path: String, val size: Long) {
        val name get() = path.substringAfterLast('/')
    }

    var downloading by mutableStateOf(false)
        private set
    var dlStatus by mutableStateOf("")
        private set

    fun dir(ctx: Context): File = File(ctx.filesDir, "models").apply { mkdirs() }

    private fun manifestFile(ctx: Context) = File(dir(ctx), "manifest.txt")

    private fun manifest(ctx: Context): Map<String, String> {
        val f = manifestFile(ctx)
        if (!f.exists()) return emptyMap()
        return f.readLines().mapNotNull {
            val i = it.indexOf('=')
            if (i > 0) it.substring(0, i) to it.substring(i + 1) else null
        }.toMap()
    }

    /** name is e.g. "embed_tokens.onnx"; resolves to the real downloaded file name. */
    fun file(ctx: Context, name: String): File = File(dir(ctx), manifest(ctx)[name] ?: name)

    fun isReady(ctx: Context): Boolean =
        manifestFile(ctx).exists() && File(dir(ctx), "tokenizer.json").exists()

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

    private fun open(url: String, have: Long = 0L): HttpURLConnection {
        val c = URL(url).openConnection() as HttpURLConnection
        c.connectTimeout = 20000
        c.readTimeout = 30000
        c.instanceFollowRedirects = true
        if (have > 0) c.setRequestProperty("Range", "bytes=$have-")
        return c
    }

    private fun listRepo(repo: String): List<Item> {
        val c = open("https://huggingface.co/api/models/$repo/tree/main?recursive=true")
        if (c.responseCode != 200) return emptyList()
        val arr = JSONArray(c.inputStream.bufferedReader().readText())
        val out = ArrayList<Item>()
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            if (o.optString("type") == "file") out.add(Item(repo, o.getString("path"), o.optLong("size", 0L)))
        }
        return out
    }

    /** Finds each model (preferring the _q4 version) and its .onnx_data companion. */
    private fun plan(): Pair<List<Item>, Map<String, String>> {
        val listings = REPOS.map { r -> try { listRepo(r) } catch (e: Exception) { emptyList() } }
        val items = ArrayList<Item>()
        val map = LinkedHashMap<String, String>()
        for (base in BASES) {
            var chosen: List<Item>? = null
            var chosenOnnx: Item? = null
            for (files in listings) {
                val re = Regex("^${base}(_q4)?\\.onnx(_data)?$")
                val m = files.filter { re.matches(it.name) }
                if (m.isEmpty()) continue
                val useQ4 = m.any { it.name.contains("_q4") }
                val pick = m.filter { it.name.contains("_q4") == useQ4 }
                val onnx = pick.firstOrNull { it.name.endsWith(".onnx") } ?: continue
                chosen = pick
                chosenOnnx = onnx
                break
            }
            if (chosen == null || chosenOnnx == null) throw IOException("Could not find $base on Hugging Face")
            items.addAll(chosen)
            map["$base.onnx"] = chosenOnnx.name
        }
        return items to map
    }

    private fun fetch(dir: File, name: String, url: String, label: String) {
        val out = File(dir, name)
        val part = File(dir, "$name.part")
        val have = if (part.exists()) part.length() else 0L
        val c = open(url, have)
        c.connect()
        val code = c.responseCode
        if (code == 416 && have > 0) {
            if (!part.renameTo(out)) throw IOException("Could not save $name")
            return
        }
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
                        dlStatus = "Downloading $label: ${done / 1048576} / ${total / 1048576} MB"
                    }
                }
            }
        }
        if (!part.renameTo(out)) throw IOException("Could not save $name")
    }

    private fun download(ctx: Context) {
        val dir = dir(ctx)
        if (!File(dir, "tokenizer.json").exists()) fetch(dir, "tokenizer.json", TOK, "tokenizer")
        dlStatus = "Finding model files..."
        val (items, map) = plan()
        for ((i, it) in items.withIndex()) {
            val out = File(dir, it.name)
            if (out.exists() && (it.size <= 0 || out.length() == it.size)) continue
            fetch(
                dir, it.name,
                "https://huggingface.co/${it.repo}/resolve/main/${it.path}",
                "${it.name} (${i + 1}/${items.size})"
            )
        }
        manifestFile(ctx).writeText(map.entries.joinToString("\n") { "${it.key}=${it.value}" })
    }
}
