package com.voicestudio.app

import org.json.JSONArray
import org.json.JSONObject
import java.text.Normalizer

/** Minimal reader for a HuggingFace BPE tokenizer.json (normalizer, pre-tokenizer, BPE, template). */
class ChatterboxTokenizer(json: String) {
    private class Added(val id: Int, val content: String, val normalized: Boolean)
    private class Seg(val text: String, val id: Int?)

    private val vocab = HashMap<String, Int>()
    private val ranks = HashMap<String, Int>()
    private val added = ArrayList<Added>()
    private var unkId: Int? = null
    private val normOps: List<JSONObject>
    private val preOps: List<JSONObject>
    private val template = ArrayList<IntArray?>() // null = the text slot

    init {
        val root = JSONObject(json)
        val model = root.getJSONObject("model")
        val v = model.getJSONObject("vocab")
        for (k in v.keys()) vocab[k] = v.getInt(k)
        val m = model.getJSONArray("merges")
        for (i in 0 until m.length()) {
            val e = m.get(i)
            val key = if (e is JSONArray) e.getString(0) + " " + e.getString(1) else e as String
            ranks[key] = i
        }
        val unk = model.optString("unk_token", "")
        if (unk.isNotEmpty()) unkId = vocab[unk]

        root.optJSONArray("added_tokens")?.let { arr ->
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                added.add(Added(o.getInt("id"), o.getString("content"), o.optBoolean("normalized", false)))
            }
            added.sortByDescending { it.content.length }
        }

        normOps = flatten(root.optJSONObject("normalizer"), "normalizers")
        preOps = flatten(root.optJSONObject("pre_tokenizer"), "pretokenizers")

        val pp = root.optJSONObject("post_processor")
        if (pp != null && pp.optString("type") == "TemplateProcessing") {
            val specials = pp.getJSONObject("special_tokens")
            val single = pp.getJSONArray("single")
            for (i in 0 until single.length()) {
                val e = single.getJSONObject(i)
                if (e.has("SpecialToken")) {
                    val id = e.getJSONObject("SpecialToken").getString("id")
                    val arr = specials.getJSONObject(id).getJSONArray("ids")
                    template.add(IntArray(arr.length()) { arr.getInt(it) })
                } else {
                    template.add(null)
                }
            }
        }
        if (template.isEmpty()) template.add(null)
    }

    private fun flatten(o: JSONObject?, key: String): List<JSONObject> {
        if (o == null) return emptyList()
        if (o.optString("type") == "Sequence") {
            val arr = o.getJSONArray(key)
            return (0 until arr.length()).flatMap { flatten(arr.getJSONObject(it), key) }
        }
        return listOf(o)
    }

    private fun normalize(s: String): String {
        var t = s
        for (op in normOps) {
            t = when (op.optString("type")) {
                "Lowercase" -> t.lowercase()
                "NFC" -> Normalizer.normalize(t, Normalizer.Form.NFC)
                "NFD" -> Normalizer.normalize(t, Normalizer.Form.NFD)
                "NFKC" -> Normalizer.normalize(t, Normalizer.Form.NFKC)
                "NFKD" -> Normalizer.normalize(t, Normalizer.Form.NFKD)
                "Replace" -> {
                    val p = op.getJSONObject("pattern")
                    val c = op.getString("content")
                    if (p.has("String")) t.replace(p.getString("String"), c)
                    else t.replace(Regex(p.getString("Regex")), c)
                }
                "Prepend" -> op.getString("prepend") + t
                "Strip" -> t.trim()
                else -> t
            }
        }
        return t
    }

    private fun pre(s: String): List<String> {
        var parts = listOf(s)
        for (op in preOps) {
            parts = parts.flatMap { p ->
                when (op.optString("type")) {
                    "Whitespace" -> Regex("(?U)\\w+|[^\\w\\s]+").findAll(p).map { it.value }.toList()
                    "WhitespaceSplit" -> p.split(Regex("(?U)\\s+")).filter { it.isNotEmpty() }
                    "Split" -> {
                        val pat = op.getJSONObject("pattern")
                        val re = if (pat.has("String")) Regex(Regex.escape(pat.getString("String")))
                        else Regex(pat.getString("Regex"))
                        if (op.optString("behavior") == "Removed") {
                            p.split(re).filter { it.isNotEmpty() }
                        } else {
                            val res = ArrayList<String>()
                            var last = 0
                            for (m in re.findAll(p)) {
                                if (m.range.first > last) res.add(p.substring(last, m.range.first))
                                if (m.value.isNotEmpty()) res.add(m.value)
                                last = m.range.last + 1
                            }
                            if (last < p.length) res.add(p.substring(last))
                            res
                        }
                    }
                    else -> listOf(p)
                }
            }
        }
        return parts
    }

    private fun bpe(piece: String): List<Int> {
        val syms = ArrayList<String>()
        piece.codePoints().forEach { syms.add(String(Character.toChars(it))) }
        while (syms.size > 1) {
            var best = -1
            var bestRank = Int.MAX_VALUE
            for (i in 0 until syms.size - 1) {
                val r = ranks[syms[i] + " " + syms[i + 1]]
                if (r != null && r < bestRank) {
                    bestRank = r
                    best = i
                }
            }
            if (best < 0) break
            syms[best] = syms[best] + syms[best + 1]
            syms.removeAt(best + 1)
        }
        return syms.mapNotNull { vocab[it] ?: unkId }
    }

    private fun split(text: String, toks: List<Added>): List<Seg> {
        if (toks.isEmpty()) return listOf(Seg(text, null))
        val res = ArrayList<Seg>()
        val sb = StringBuilder()
        var i = 0
        while (i < text.length) {
            val t = toks.firstOrNull { text.startsWith(it.content, i) }
            if (t != null) {
                if (sb.isNotEmpty()) {
                    res.add(Seg(sb.toString(), null))
                    sb.clear()
                }
                res.add(Seg("", t.id))
                i += t.content.length
            } else {
                sb.append(text[i])
                i++
            }
        }
        if (sb.isNotEmpty()) res.add(Seg(sb.toString(), null))
        return res
    }

    fun encode(text: String): LongArray {
        val ids = ArrayList<Int>()
        for (s1 in split(text, added.filter { !it.normalized })) {
            if (s1.id != null) {
                ids.add(s1.id)
                continue
            }
            val n = normalize(s1.text)
            for (s2 in split(n, added.filter { it.normalized })) {
                if (s2.id != null) ids.add(s2.id)
                else for (p in pre(s2.text)) ids.addAll(bpe(p))
            }
        }
        val out = ArrayList<Long>()
        for (t in template) {
            if (t == null) ids.forEach { out.add(it.toLong()) }
            else t.forEach { out.add(it.toLong()) }
        }
        return out.toLongArray()
    }
}
