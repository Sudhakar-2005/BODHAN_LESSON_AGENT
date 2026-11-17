# Demonstration source provenance

Retrieved 2026-11-17. Both files are public NCERT (Government of India,
National Council of Educational Research and Training) educational PDFs from
the official `ncert.nic.in` domain. Originals are stored byte-for-byte as
downloaded; the pipeline never modifies them (it rasterises pages into its
own work directory).

## 1. English source — NCERT Class 9 Mathematics exemplar

- File: `ncert-class9-maths-ch6-lines-and-angles-official.pdf`
- Title: Exemplar Problems in Mathematics, Class IX — Chapter 6: Lines and Angles
- Publisher: NCERT, New Delhi
- Language: English
- Source URL: https://ncert.nic.in/pdf/publication/exemplarproblem/classIX/mathematics/ieep206.pdf
- Retrieval date: 2026-11-17
- Pages: 9 (within the 5–30 supported range)
- SHA-256: see `sha256sum.txt` in this directory

## 2. Indian-language source — NCERT Class 9 Mathematics (Hindi medium)

- File: `ncert-class9-maths-ch5-rekha-aur-kon-hindi-official.pdf`
- Title: गणित मंजरी / NCERT Class 9 Mathematics (Hindi medium) — रेखा और कोण
- Publisher: NCERT, New Delhi
- Language: Hindi
- Source URL: https://ncert.nic.in/textbook/pdf/ghmh105.pdf
- Retrieval date: 2026-11-17
- Pages: 14 (within the 5–30 supported range)
- Note: the PDF's embedded text layer is mis-encoded (no usable Devanagari
  Unicode). This is intentional for the demonstration: the pipeline never
  reads the text layer; it rasterises each page and uses Bodhan Indic-OCR
  exclusively, so no embedded text is ever used as OCR input.
- SHA-256: see `sha256sum.txt` in this directory
