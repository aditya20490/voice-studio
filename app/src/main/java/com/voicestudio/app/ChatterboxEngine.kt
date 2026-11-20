package com.voicestudio.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/** Runs the Chatterbox multilingual ONNX pipeline fully on-device. */
class ChatterboxEngine private constructor(private val ctx: Context) {
    companion object {
        private const val START = 6561L
        private const val STOP = 6562L
        private const val LAYERS = 30
        private const val HEADS = 16L
        private const val HEAD_DIM = 64L
        private const val MAX_NEW = 256
        private const val PENALTY = 1.2f
        private const val SR = 24000

        @Volatile
        private var inst: ChatterboxEngine? = null

        fun get(ctx: Context): ChatterboxEngine =
            inst ?: synchronized(this) {
                inst ?: ChatterboxEngine(ctx.applicationContext).also { inst = it }
            }
    }

    private class Cond(
        val condEmb: FloatArray, val condShape: LongArray,
        val promptToken: LongArray,
        val xvec: FloatArray, val xShape: LongArray,
        val feat: FloatArray, val featShape: LongArray
    )

    var status by mutableStateOf("Ready")
        private set
    var busy by mutableStateOf(false)
        private set
    var result by mutableStateOf<File?>(null)
        private set

    @Volatile
    private var cancelled = false

    private val env = OrtEnvironment.getEnvironment()
    private var tok: ChatterboxTokenizer? = null
    private var embed: OrtSession? = null
    private var lm: OrtSession? = null
    private var dec: OrtSession? = null

    private val kvNames = (0 until LAYERS).flatMap { l ->
        listOf("past_key_values.$l.key", "past_key_values.$l.value")
    }

    fun cancel() {
        cancelled = true
    }

    fun start(text: String, voice: Voice, exaggeration: Float) {
        if (busy) return
        busy = true
        cancelled = false
        result = null
        Thread {
            try {
                result = run(text, voice, exaggeration)
                status = "Done"
            } catch (e: Throwable) {
                status = "Error: ${e.message ?: e.javaClass.simpleName}"
            } finally {
                busy = false
            }
        }.start()
    }

    private fun opts() = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(4)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
    }

    private fun path(name: String) = Models.file(ctx, name).absolutePath

    private fun load() {
        if (tok == null) tok = ChatterboxTokenizer(Models.file(ctx, "tokenizer.json").readText())
        if (embed == null) embed = env.createSession(path("embed_tokens.onnx"), opts())
        if (lm == null) lm = env.createSession(path("language_model.onnx"), opts())
        if (dec == null) dec = env.createSession(path("conditional_decoder.onnx"), opts())
    }

    private fun ft(d: FloatArray, shape: LongArray) = OnnxTensor.createTensor(env, FloatBuffer.wrap(d), shape)
    private fun lt(d: LongArray, shape: LongArray) = OnnxTensor.createTensor(env, LongBuffer.wrap(d), shape)

    private fun floats(v: OnnxValue): Pair<FloatArray, LongArray> {
        val t = v as OnnxTensor
        val fb = t.floatBuffer
        val a = FloatArray(fb.remaining())
        fb.get(a)
        return a to t.info.shape
    }

    private fun longs(v: OnnxValue): LongArray {
        val lb = (v as OnnxTensor).longBuffer
        val a = LongArray(lb.remaining())
        lb.get(a)
        return a
    }

    private fun encodeReference(audio: FloatArray): Cond {
        val s = env.createSession(path("speech_encoder.onnx"), opts())
        try {
            val t = ft(audio, longArrayOf(1, audio.size.toLong()))
            try {
                s.run(mapOf("audio_values" to t)).use { r ->
                    val (ce, cs) = floats(r.get(0))
                    val pt = longs(r.get(1))
                    val (xv, xs) = floats(r.get(2))
                    val (pf, ps) = floats(r.get(3))
                    return Cond(ce, cs, pt, xv, xs, pf, ps)
                }
            } finally {
                t.close()
            }
        } finally {
            s.close()
        }
    }

    private fun generate(ids: LongArray, exag: Float, cond: Cond, label: String): LongArray {
        val embedS = embed!!
        val lmS = lm!!
        val exagT = ft(floatArrayOf(exag), longArrayOf(1))
        val generated = ArrayList<Long>()
        generated.add(START)
        var nextIds = ids
        var nextPos = LongArray(ids.size) { if (ids[it] >= START) 0L else it - 1L }
        var maskLen = 0
        var past: List<OnnxTensor> = kvNames.map { ft(FloatArray(0), longArrayOf(1, HEADS, 0, HEAD_DIM)) }
        var ownPast = true
        var last: OrtSession.Result? = null
        try {
            for (i in 0 until MAX_NEW) {
                if (cancelled) break
                status = "$label - step ${i + 1}"

                val idsT = lt(nextIds, longArrayOf(1, nextIds.size.toLong()))
                val posT = lt(nextPos, longArrayOf(1, nextPos.size.toLong()))
                val (emb0, eShape) = embedS.run(
                    mapOf("input_ids" to idsT, "position_ids" to posT, "exaggeration" to exagT)
                ).use { floats(it.get(0)) }
                idsT.close()
                posT.close()

                val hidden = eShape[2]
                var embeds = emb0
                var seq = eShape[1]
                if (i == 0) {
                    embeds = cond.condEmb + emb0
                    seq += cond.condShape[1]
                    maskLen = seq.toInt()
                } else {
                    maskLen += 1
                }

                val inputs = HashMap<String, OnnxTensor>()
                val embT = ft(embeds, longArrayOf(1, seq, hidden))
                val maskT = lt(LongArray(maskLen) { 1L }, longArrayOf(1, maskLen.toLong()))
                inputs["inputs_embeds"] = embT
                inputs["attention_mask"] = maskT
                kvNames.forEachIndexed { j, n -> inputs[n] = past[j] }

                val res = lmS.run(inputs)
                embT.close()
                maskT.close()

                val logitsT = res.get(0) as OnnxTensor
                val ls = logitsT.info.shape
                val vocab = ls[2].toInt()
                val off = ((ls[1] - 1) * vocab).toInt()
                val fb = logitsT.floatBuffer
                val scores = FloatArray(vocab) { fb.get(off + it) }
                for (t in generated.toSet()) {
                    val k = t.toInt()
                    if (k in 0 until vocab) {
                        val sc = scores[k]
                        scores[k] = if (sc < 0) sc * PENALTY else sc / PENALTY
                    }
                }
                var best = 0
                for (k in 1 until vocab) if (scores[k] > scores[best]) best = k
                val next = best.toLong()
                generated.add(next)

                val newPast = (0 until kvNames.size).map { res.get(it + 1) as OnnxTensor }
                if (ownPast) past.forEach { it.close() }
                ownPast = false
                last?.close()
                last = res
                past = newPast

                if (next == STOP) break
                nextIds = longArrayOf(next)
                nextPos = longArrayOf(i + 1L)
            }
        } finally {
            last?.close()
            if (ownPast) past.forEach { it.close() }
            exagT.close()
        }
        val toks = generated.drop(1).toMutableList()
        if (toks.isNotEmpty() && toks.last() == STOP) toks.removeAt(toks.size - 1)
        return cond.promptToken + toks.toLongArray()
    }

    private fun decode(tokens: LongArray, c: Cond): FloatArray {
        val a = lt(tokens, longArrayOf(1, tokens.size.toLong()))
        val b = ft(c.xvec, c.xShape)
        val d = ft(c.feat, c.featShape)
        try {
            return dec!!.run(
                mapOf("speech_tokens" to a, "speaker_embeddings" to b, "speaker_features" to d)
            ).use { floats(it.get(0)).first }
        } finally {
            a.close()
            b.close()
            d.close()
        }
    }

    private fun chunks(text: String): List<String> {
        val out = ArrayList<String>()
        val sentences = text.split(Regex("(?<=[.!?\u0964])\\s+|\\n+")).map { it.trim() }.filter { it.isNotEmpty() }
        for (s in sentences) {
            var cur = StringBuilder()
            for (p in s.split(Regex("(?<=[,;:])\\s+"))) {
                if (cur.isNotEmpty() && cur.length + p.length > 200) {
                    out.add(cur.toString())
                    cur = StringBuilder()
                }
                if (cur.isNotEmpty()) cur.append(' ')
                cur.append(p)
            }
            if (cur.isNotEmpty()) out.add(cur.toString())
        }
        return out
    }

    private fun langOf(s: String) = if (s.any { it in '\u0900'..'\u097F' }) "hi" else "en"

    private fun run(text: String, voice: Voice, exag: Float): File {
        status = "Loading models (first time takes a while)..."
        load()
        status = "Reading your voice..."
        val audio = AudioLoader.loadAll(voice.clips, 10)
        if (audio.size < SR) error("Voice recording is too short")
        val cond = encodeReference(audio)

        val parts = chunks(text)
        if (parts.isEmpty()) error("Type some text first")
        val t = tok!!
        val pcm = ArrayList<FloatArray>()
        parts.forEachIndexed { i, part ->
            if (cancelled) error("Cancelled")
            val ids = t.encode("[${langOf(part)}]$part")
            val tokens = generate(ids, exag, cond, "Part ${i + 1} of ${parts.size}")
            if (cancelled) error("Cancelled")
            status = "Part ${i + 1} of ${parts.size} - making audio..."
            pcm.add(decode(tokens, cond))
            pcm.add(FloatArray(SR * 3 / 10))
        }

        val total = pcm.sumOf { it.size }
        val shorts = ShortArray(total)
        var k = 0
        for (arr in pcm) for (x in arr) shorts[k++] = (x.coerceIn(-1f, 1f) * 32767f).toInt().toShort()

        val dir = File(ctx.filesDir, "outputs").apply { mkdirs() }
        val f = File(dir, "speech_${System.currentTimeMillis()}.wav")
        writeWav(f, shorts)
        return f
    }

    private fun writeWav(f: File, pcm: ShortArray) {
        val bb = ByteBuffer.allocate(44 + pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray())
        bb.putInt(36 + pcm.size * 2)
        bb.put("WAVE".toByteArray())
        bb.put("fmt ".toByteArray())
        bb.putInt(16)
        bb.putShort(1)
        bb.putShort(1)
        bb.putInt(SR)
        bb.putInt(SR * 2)
        bb.putShort(2)
        bb.putShort(16)
        bb.put("data".toByteArray())
        bb.putInt(pcm.size * 2)
        for (s in pcm) bb.putShort(s)
        f.writeBytes(bb.array())
    }
}
