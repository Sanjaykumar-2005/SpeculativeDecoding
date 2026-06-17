"""
prompts.py
==========
OCR-style extraction prompts for the speculative-decoding benchmark — wired to
the COMPANY'S PRODUCTION prompt for the L&T OCR AI Solution.

Real OCR pipelines first turn a scanned document image into raw text (via a
vision/OCR engine), then ask an LLM to pull out structured fields. This POC is
about the *decoding* technique, so we simulate the second stage: we embed a
chunk of already-OCR'd invoice text and ask the model to extract a single field.

The single-keyword extraction path (`get_openai_extraction_prompt`) and the
shared `get_invoice_validation_rules()` block below are taken from the company's
`prompts.py` so the benchmark and the Streamlit UI run on the *actual* prompt
that will be deployed on the H200 (Qwen 3B draft + 32B target). The validation
rules are long (~2k tokens) but that is fine for speed: the prompt is processed
once in the prefill, and BOTH the target-alone baseline and the speculative run
pay the same prefill cost, so the speedup comparison stays fair.

What this enables you to SEE in the demo:
  * the NUMERIC OUTPUT rule in action — the model should turn "₹ 15,871.00" in
    the document into the plain string "15871.00";
  * the Vendor-vs-Consignee rule — Vendor GSTIN must be the vendor's, never the
    L&T / consignee GSTIN that also appears in the document.
"""

import json

# ---------------------------------------------------------------------------
# A realistic block of "OCR output" from a scanned L&T vendor invoice.
# It deliberately contains BOTH a vendor and an L&T (consignee) GSTIN/PAN, a
# 64-char IRN, bank/IFSC details, and a rupee-formatted total, so the company
# validation rules below actually have something to act on.
# ---------------------------------------------------------------------------
SAMPLE_INVOICE = """\
==================== TAX INVOICE (e-Invoice) ====================
IRN      : a4f7c2b9e1d63089a4f7c2b9e1d63089a4f7c2b9e1d63089a4f7c2b9e1d63089
Ack No   : 112410012345678          Ack Date : 12-06-2026

Supplier / Vendor:
  Sterling Office Supplies Pvt Ltd
  24 Anna Salai, Chennai, Tamil Nadu - 600002
  GSTIN : 33AABCS1429B1ZX
  PAN   : AABCS1429B

Consignee / Billed To:
  Larsen & Toubro Limited (L&T)
  Mount Poonamallee Road, Manapakkam, Chennai - 600089
  GSTIN : 33AAACL0140P1ZV
  PAN   : AAACL0140P

Invoice No   : INV-2026-004871
Invoice Date : 12-06-2026
PO Number    : PO-778451

Bank Details:
  Bank   : Indian Overseas Bank
  A/C No : 015702000012345
  IFSC   : IOBA0000157
------------------------------------------------------------------
Description            Qty     Rate      Amount
A4 Paper (500 sheets)   20    320.00    6,400.00
Ballpoint Pens (box)    15    150.00    2,250.00
Stapler HD-45           10    480.00    4,800.00
------------------------------------------------------------------
Subtotal                              13,450.00
CGST 9%                                 1,210.50
SGST 9%                                 1,210.50
TOTAL AMOUNT                         ₹ 15,871.00
==================================================================
"""

# Each task is a FIELD KEYWORD to extract. These are fed to the company's
# single-keyword extraction prompt below. This is the COMPREHENSIVE set of
# fields found in an L&T vendor invoice — every meaningful header/party/tax/bank
# field. (In the UI you can deselect any you don't want; running all of them is
# 3 model passes per field, so it takes a few minutes on the laptop.)
OCR_TASKS = [
    # --- header / identity ---
    "Invoice Number",
    "Invoice Date",
    "PO Number",
    "IRN",
    # --- vendor (supplier) ---
    "Vendor Name",
    "Vendor Address",
    "Vendor GSTIN",
    "Vendor PAN",
    # --- consignee (buyer / L&T) ---
    "Consignee Name",
    "Consignee GSTIN",
    # --- line-item / tax ---
    "HSN/SAC Code",
    "Taxable Value",
    "IGST Amount",
    "Total Amount",
    # --- bank ---
    "Bank Name",
    "Bank Account Number",
    "IFSC Code",
]

# ---------------------------------------------------------------------------
# A REAL OCR-engine output (HTML tables, exactly as produced by the pipeline).
# This is messy on purpose — it exercises the validation/OCR-correction rules:
#   * Vendor GST "07ADDP06628E1ZS" and stated "PAN NO. ADDPC8628E" disagree (OCR).
#   * L&T (consignee) GST "24AAACLO140P7ZJ" has a letter 'O' where the fixed L&T
#     PAN "AAACL0140P" needs a zero -> the model must NOT return this as Vendor.
#   * Total is ambiguous: TOTAL 265332.28 vs GROSS Total 3,13,092.10 (Indian
#     digit grouping) -> "Total Amount" should be the gross 313092.10 (numeric).
#   * There is NO IRN -> the IRN task should return NOT FOUND.
# ---------------------------------------------------------------------------
REAL_OCR_CHAUDHARY = (
    "<p>Bill 26/001950</p> <p>ORIGINAL INVOICE</p> <p><b>Tax Invoice</b></p> "
    "<table> <tr> <td> <b>CHAUDHARY TRANSPORT COMPANY</b><br/> 10, TRANSPORT "
    "CENTRE MAIN ROHTAK ROAD, PUNJABI BAGH, NEW DELHI-110035.<br/> GST NO. : "
    "07ADDP06628E1ZS<br/> STATE NAME : DELHI, CODE : 07<br/> PAN NO. ADDPC8628E"
    "<br/> E-Mail : ctc.vijender@gmail.com/ctc v@yahoo.com<br/> Buyer </td> "
    "<td>Invoice No.<br/>CTC/2627/1284/01</td> <td>Dated<br/>02-Jun-26</td> </tr> "
    "<tr> <td>PERIOD<br/>Mar-26</td> <td>Reverse Charge (Y/N)<br/>No</td> </tr> "
    "<tr> <td>Buyer's Order No.<br/>LE/LE21M884/WOD/22/000050</td> <td>Dated</td> "
    "</tr> <tr> <td> <b>Larsen & Toubro Ltd Construction</b><br/> MAHSR C4 SECTION "
    "5, SH-161, NEAR SAPA ROAD JUNCTION, VILLAGE MANGROL, KARJAN TALUK VADODARA "
    "GUJARAT-391240<br/> GST NO- 24AAACLO140P7ZJ<br/> STATE NAME : GUJARAT , CODE "
    ": 24 </td> <td> <b>Larsen & Toubro Ltd Construction</b><br/> MAHSR C4 SECTION "
    "5, SH-161, NEAR SAPA ROAD JUNCTION, VILLAGE MANGROL, KARJAN TALUK VADODARA "
    "GUJARAT-391240<br/> GST NO- 24AAACLO140P7ZJ<br/> STATE NAME : GUJARAT , CODE "
    ": 24 </td> </tr> </table> <table> <tr> <th>SI No</th> <th>Description</th> "
    "<th>HSN/SAC</th> <th>UOM</th> <th>Quantity</th> <th>Rate</th> <th>Amount</th> "
    "</tr> <tr> <td>1</td> <td>Working of Tyre Mounted 40T Crane No- GJ02BS1284</td> "
    "<td>997313</td> <td>HOURS</td> <td>312.00</td> <td>833.33</td> <td>259998.96"
    "</td> </tr> <tr> <td>2</td> <td>Overtime Charges</td> <td>997313</td> "
    "<td>HOURS</td> <td>4.00</td> <td>583.331</td> <td>2333.32</td> </tr> <tr> "
    "<td>3</td> <td>Accommodation</td> <td>997313</td> <td>L.S.</td> <td>0.50</td> "
    "<td>6000.00</td> <td>3000.00</td> </tr> <tr> <td><b>TOTAL</b></td> "
    "<td><b>265332.28</b></td> </tr> <tr> <td><b>IGST</b></td> <td><b>18%</b></td> "
    "<td><b>47759.81</b></td> </tr> <tr> <td><b>GROSS Total</b></td> "
    "<td><b>3,13,092.10</b></td> </tr> </table> <p>Total Amount Chargeable (in "
    "words) : Indian Rupees<br/> <b>THREE LAKH THIRTEEN THOUSAND NINETY TWO RUPEES "
    "AND TEN PAISE ONLY</b></p> <table> <tr> <th>HSN/SAC</th> <th>Taxable Value</th> "
    "<th>IGST</th> <th>Total Tax Amount</th> </tr> <tr> <th>Rate</th> <th>Amount</th> "
    "</tr> <tr> <td>997313</td> <td>265332.28</td> <td>18%</td> <td>47759.81</td> "
    "<td>47,759.81</td> </tr> <tr> <td><b>Total</b></td> <td><b>265332.28</b></td> "
    "<td></td> <td><b>47759.81</b></td> </tr> </table> <p>Tax Amount (in words) "
    ":Indian Rupees<br/> <b>FORTY SEVEN THOUSAND SEVEN HUNDRED FIFTY NINE RUPEES "
    "AND EIGHTY ONE PAISE ONLY</b></p> <table> <tr> <td> <b>BANK DETAILS</b><br/> "
    "BANK NAME : PUNJAB NATIONAL BANK<br/> ACCOUNT NO : 0605009300145700<br/> IFSC "
    "CODE : PUNB0060500<br/> BANK ADDRESS : RAJOURI GARDEN, NEW DELHI -110027 </td> "
    "<td> for M/s Chaudhary Transport Company<br/> <br/> Authorized Signature<br/> "
    "M/s. Signatory </td> </tr> </table>"
)

# Documents selectable in the Streamlit UI. The first is the clean synthetic
# invoice; the second is the real (messy) OCR-engine HTML output above.
SAMPLE_DOCUMENTS = {
    "Synthetic L&T invoice (clean)": SAMPLE_INVOICE,
    "Real OCR — Chaudhary Transport (HTML)": REAL_OCR_CHAUDHARY,
}


# ===========================================================================
# COMPANY PRODUCTION PROMPT (verbatim logic from the L&T OCR AI Solution)
# ===========================================================================
def get_invoice_validation_rules() -> str:
    """
    Common invoice validation rules for GSTIN, PAN, IFSC, Bank Account, IRN
    extraction. Appended to every extraction prompt for consistent validation.
    (Source: company prompts.py.)
    """
    return """

    ================================================================================
    INVOICE VALIDATION RULES
    ================================================================================

    **DATA EXTRACTION PRIORITY:**
    If the extracted content contains invoice-related data:
    1. First, search in the **"Extracted Fields"** section
    2. If not found, look in the **"Extracted Tables"** section
    3. If still not found, check the **OCR Content** block as fallback
    Priority order: Extracted Fields -> Extracted Tables -> OCR Content

    **ENTITY CLARIFICATION:**
    - The Consignee/Buyer is typically "L&T" (Larsen & Toubro)
    - The Vendor is the service provider or seller
    - DO NOT interchange Vendor and Consignee information
    - If Vendor GSTN is missing, return "NOT FOUND". DO NOT use Consignee/L&T GSTN

    --------------------------------------------------------------------------------
    GSTIN FORMAT & RULES
    --------------------------------------------------------------------------------
    - Length: 15 characters
    - Format: 2 digits + 10-character PAN + 1 digit + 'Z' + 1 digit
    - Example: 33ABCDE1234F1Z5
    - The 5th character must be an alphabet
    - If the 5th character is '0' (zero), replace it with 'O' (capital letter O)
    - Example: 21APOPB6452J1ZY (correct) NOT 21AP0PB6452J1ZY (incorrect)

    --------------------------------------------------------------------------------
    COMPANY / L&T / CONSIGNEE GSTIN KEYWORD MAPPING RULES
    --------------------------------------------------------------------------------
    - "L&T GSTIN" / "Company GSTIN" / "Consignee GSTIN" may not be explicitly labeled
    - If text contains "To - LARSEN & TOUBRO" or "L&T"
      -> The GSTIN near this section is the Company GSTIN
    - L&T GSTIN contains a fixed PAN value in characters 3 to 12
    - L&T PAN = AAACL0140P
    - Check all GSTINs: if any GSTIN contains "AAACL0140P" in position 3-12,
      select it as the Company / L&T GSTIN
      Example: 27AAACL0140P5ZF -> correct (contains AAACL0140P)
    - If multiple GSTINs exist:
      1. Prefer GSTIN starting with "27AAA"
      2. Else prefer GSTIN closest to "L&T" / "Consignee" keywords
      3. Else return the first valid GSTIN

    --------------------------------------------------------------------------------
    PAN FORMAT & EXTRACTION RULES
    --------------------------------------------------------------------------------
    - Length: 10 characters
    - Format: 5 uppercase letters + 4 digits + 1 uppercase letter
    - Example: ABCDE1234F
    **Vendor PAN Extraction:**
    1. First search for Vendor PAN explicitly in: Extracted Fields -> Extracted Tables -> OCR Content
    2. If NOT explicitly present, extract from Vendor GSTIN (characters 3-12)
    3. Vendor GSTIN must belong ONLY to the Vendor section
    **DO NOT use for Vendor PAN:** Customer GSTIN, Billed To / Shipped To GSTIN,
    L&T GSTIN or any GSTIN belonging to Larsen & Toubro Limited.
    If both Vendor PAN and L&T PAN are present, always pick the PAN associated with Vendor Name.

    --------------------------------------------------------------------------------
    IFSC CODE FORMAT & RULES
    --------------------------------------------------------------------------------
    - Length: 11 characters
    - Format: 4 letters (bank code) + 0 + 6 digits (branch code)
    - Example: IOBA0000124, CNRB0001234
    - If IFSC starts with BARB, 5th character is always '0' (zero), not 'O'
    - Correct errors like '8' instead of 'B' or 'I' instead of '1'

    --------------------------------------------------------------------------------
    BANK ACCOUNT NUMBER RULES
    --------------------------------------------------------------------------------
    - Usually 9-18 digits long
    - Do NOT confuse with cheque number or MICR code
    - Cheque/MICR numbers typically appear at the end of cheque content and are longer

    --------------------------------------------------------------------------------
    IRN (Invoice Reference Number) RULES
    --------------------------------------------------------------------------------
    - A 64-character hexadecimal string
    - Contains only digits (0-9) and lowercase letters (a-f)
    - No spaces
    - Look for labels: IRN, IRN No, Invoice Reference Number, e-Invoice IRN
    - If broken into multiple lines in OCR, join to form continuous 64-char string
    - Do NOT confuse with: Ack No, EWB No, Invoice No, Dispatch No

    --------------------------------------------------------------------------------
    ITEM CODE EXTRACTION RULES
    --------------------------------------------------------------------------------
    If the requested field is Material Code:
    - Extract only from columns like Material Code, Item Code, Product Code, or Material Number.
    - Do NOT extract from Sl No, Serial Number, HSN Code, SAC Code, or Quantity.
    - Material Code is usually a long numeric/alphanumeric value (example: 190000000006931).
    - Ignore short values like 1, 2, 3 because they are usually serial numbers.
    - If both Material Code and HSN exist, always return the Material Code.

    --------------------------------------------------------------------------------
    OCR CORRECTION RULES
    --------------------------------------------------------------------------------
    For PAN, GSTIN, and IFSC codes, correct potential OCR misreads:
    - '1' <-> 'I' (one vs capital i)
    - '8' <-> 'B' (eight vs capital b)
    - '0' <-> 'O' (zero vs capital o)
    - '2' <-> 'Z' (two vs capital z)
    - '5' <-> 'S' (five vs capital s)
    Apply corrections only when the result violates the expected format pattern.

    --------------------------------------------------------------------------------
    GST VALUE RULES
    --------------------------------------------------------------------------------
    - Do NOT calculate GST values (IGST, CGST, SGST) from percentages
    - Extract only explicitly stated values

    ================================================================================
    STRICT NUMERIC OUTPUT RULES - MANDATORY FOR ALL AMOUNT FIELDS
    ================================================================================
    ALL monetary/amount fields MUST contain ONLY numeric values.
    STRICTLY FORBIDDEN in output: rupee symbol, Rs / Rs. / INR, $, USD, EUR or any
    currency text, commas (,) used as thousand separators, trailing /- or spaces.
    REQUIRED FORMAT - plain numeric string only:
        INVALID (NEVER return): "10,169.88", "Rs. 25,000/-", "INR 1,234.56"
        VALID   (ALWAYS return): "10169.88", "25000", "1234.56"
    CONVERSION EXAMPLES:
        1,50,000.00 -> 150000.00 ; Rs. 25,000/- -> 25000 ; INR 1,234.56 -> 1234.56
        $500.00 -> 500.00
    ================================================================================

    --------------------------------------------------------------------------------
    SPECIAL FIELD RULES
    --------------------------------------------------------------------------------
    - If keyword is "Legal Name", use value of "Beneficiary" if Legal Name not found
    - For dates: maintain original format unless otherwise specified

    ================================================================================
    """


def get_field_rules(keyword: str) -> str:
    """
    FIELD-TARGETED compact rules for the lite prompt: return ONLY the validation
    guidance relevant to the requested field. Appending the whole ~2k-token rule
    set to every field distracts a small (1.5B) model — e.g. GSTIN/PAN/amount
    rules pull its attention away when all you asked for is the invoice number.
    Giving each field just its own few lines keeps the prompt short and focused,
    which markedly improves small-model accuracy. The full block (mode="full")
    is for the 32B target on the H200.
    """
    k = keyword.lower()
    entity = ("Vendor = supplier/seller; Consignee/Buyer = L&T (Larsen & Toubro). "
              "When a Vendor field is asked, NEVER return the L&T/Consignee value. "
              "L&T's PAN is AAACL0140P.")
    ocr = ("Fix OCR confusions only to satisfy the format: "
           "O<->0, I<->1, B<->8, S<->5, Z<->2.")
    is_lt = ("consignee" in k or "l&t" in k or "buyer" in k)

    # --- order matters: specific checks BEFORE generic ones ----------------
    if "date" in k:
        return ("\nThe Invoice Date is the date printed for THIS invoice (near "
                "'Invoice No.'/'Dated'), NOT the period or any other date. Keep the "
                "printed format.")
    if "invoice" in k:           # invoice number (date already handled above)
        return ("\nThe Invoice Number is the value labelled 'Invoice No.' — NOT 'Bill', "
                "NOT 'Buyer's Order No.', NOT 'Ack No.', NOT any date. "
                "Return it exactly as printed.")
    if "po" in k or "order" in k:
        return ("\nThe PO Number is the code printed right after 'Buyer's Order No.' / "
                "'PO No.' / 'P.O.'. It is an alphanumeric code that MAY contain slashes "
                "and letters. Return it exactly as printed (do not return NOT FOUND if "
                "such a code is present).")
    if "irn" in k:
        return ("\nIRN = a 64-character hexadecimal string (0-9, a-f), labelled 'IRN'. "
                "If there is no IRN in the text, return NOT FOUND.")
    # Tax-amount check MUST come before the GSTIN check: "gst" is a substring of
    # "igst"/"cgst"/"sgst", which would otherwise misroute these to GSTIN.
    if "igst" in k or "cgst" in k or "sgst" in k:
        return ("\nReturn the stated tax amount (digits only, no symbol/commas). Do NOT "
                "compute it from a percentage; use the amount printed in the document.")
    if "gstin" in k or "gst" in k:
        if is_lt:
            return (f"\nReturn the Consignee / L&T GSTIN: the 15-char GSTIN that "
                    f"contains AAACL0140P in characters 3-12 (L&T's PAN). "
                    f"GSTIN = 2 digits + 10-char PAN + 1 digit + 'Z' + 1 char.\n{ocr}")
        return (f"\n{entity}\nReturn the VENDOR's GSTIN (in the supplier block), 15 "
                f"chars = 2 digits + 10-char PAN + 1 digit + 'Z' + 1 char. A GSTIN "
                f"containing AAACL0140P (chars 3-12) is L&T's, NOT the vendor's.\n{ocr}")
    if "pan" in k:
        return ("\nReturn the value printed after 'PAN' / 'PAN NO.' in the supplier "
                "block (a 10-character code: 5 letters, 4 digits, 1 letter), exactly "
                "as written. It is NOT 'AAACL0140P' (that is L&T's). If no PAN is "
                "printed, take characters 3-12 of the Vendor GSTIN.")
    if "ifsc" in k:
        return f"\nIFSC = 11 chars: 4 letters + '0' + 6 digits.\n{ocr}"
    if "account" in k:
        return ("\nBank Account Number = the 9-18 digit account number in the BANK "
                "DETAILS block; NOT the cheque/MICR/IFSC.")
    if "bank" in k:
        return ("\nReturn the Bank Name printed in the BANK DETAILS block "
                "(e.g. after 'BANK NAME :').")
    if "hsn" in k or "sac" in k:
        return ("\nReturn the HSN/SAC code from the items table — a 6-8 digit code. "
                "If several rows share one code, return that code.")
    if "address" in k:
        return ("\nReturn the VENDOR/supplier's postal address (the supplier block at "
                "the top), NOT L&T's / the consignee's address.")
    if "name" in k:
        if is_lt:
            return ("\nReturn the Consignee / Buyer name — the party the goods are "
                    "billed to (usually 'Larsen & Toubro' / L&T).")
        return ("\nReturn the Vendor/Supplier name: the company at the VERY TOP of the "
                "invoice that issues it and whose GST/PAN/bank details are listed "
                "(here a transport company). Ignore the word 'Buyer' near it. It is "
                "NOT 'Larsen & Toubro' / L&T (that party is the consignee/buyer).")
    if "taxable" in k or "subtotal" in k:
        return ("\nReturn the Taxable Value / pre-tax subtotal as digits only — no "
                "currency symbol/words, no commas. Write it like 265332.28")
    if "amount" in k or "total" in k or "value" in k:
        return ("\nReturn the numeric value using digits and a single decimal point "
                "only — no letters, no currency symbol/words, no commas. Prefer the "
                "GROSS / grand total if present. Do not compute taxes; use the stated "
                "value. Write it like 313092.10")
    return f"\n{entity}"


def get_openai_extraction_prompt(keyword: str, text: str, mode: str = "lite") -> str:
    """
    Single-keyword extraction prompt with GST, PAN, IFSC validation rules.
    (Source: company prompts.py — `get_openai_extraction_prompt`.)

    mode="full" -> the verbatim ~2k-token company validation block (production,
                   tuned for a 32B target).
    mode="lite" -> base instruction + ONLY the field-relevant rules (laptop-tuned
                   for the 0.5B/1.5B pair; see get_field_rules).
    """
    base_prompt = f"""Extract the value for '{keyword}' from the text: {text}.
Respond only with the value and nothing else.
If value is not present, give output as NOT FOUND."""

    if mode == "full":
        return base_prompt + get_invoice_validation_rules()
    return base_prompt + get_field_rules(keyword)


# ===========================================================================
# Chat-template wrapper used by the benchmark + UI
# ===========================================================================
# System role for the Qwen instruct chat template. The heavy lifting (the field
# instruction + the document text + the validation rules) goes in the user turn
# via get_openai_extraction_prompt, exactly mirroring how the company prompt is
# assembled before being sent to the model.
SYSTEM_PROMPT = (
    "You are an expert document information extraction specialist. "
    "You read OCR-processed document text and return only the requested field value."
)


def build_chat_prompt(tokenizer, task: str, document: str = SAMPLE_INVOICE,
                      mode: str = "lite") -> str:
    """
    Wrap the company single-keyword extraction prompt in Qwen's chat template.

    `task` is a FIELD KEYWORD (e.g. "Vendor GSTIN"). We build the company's
    extraction prompt for that keyword over `document`, then apply the official
    chat template. `mode` picks the validation block: "lite" (compact, default —
    works on the 1.5B) or "full" (the verbatim company prompt, for the 32B target).

    Using apply_chat_template matters: instruct models are trained with specific
    role tags; feeding raw text instead would hurt both quality AND the
    draft/target agreement that speculative decoding depends on.
    """
    user = get_openai_extraction_prompt(task, document, mode=mode)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    # add_generation_prompt=True appends the assistant tag so the model starts
    # generating the answer immediately.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ===========================================================================
# OPEN-ENDED "extract everything" prompt
# ===========================================================================
# Instead of asking for a fixed list of fields one at a time, this asks the model
# to read the document and return EVERY detail it finds as one flat JSON object.
# Nothing is enumerated/limited — the model decides what fields are present.
_EXTRACT_ALL_LITE_RULES = (
    "\nFormatting rules:\n"
    "- Amounts/totals/taxes: digits only — no currency symbol, no commas (e.g. 313092.10).\n"
    "- Keep GSTIN (15 chars), PAN (10 chars), IFSC (11 chars) exactly; fix only obvious\n"
    "  OCR slips needed to satisfy the format (O<->0, I<->1, B<->8, S<->5, Z<->2).\n"
    "- For VENDOR/supplier fields use the supplier's values, NOT Larsen & Toubro / L&T\n"
    "  (that party is the consignee/buyer). L&T's PAN is AAACL0140P.\n"
    "- Use clear snake_case or Title-Case keys; do not invent fields that are absent."
)

EXTRACT_ALL_SYSTEM = (
    "You are an expert invoice data-extraction engine. You read OCR-processed "
    "document text and return all of its details as a single JSON object."
)


def build_extract_all_prompt(tokenizer, document: str = SAMPLE_INVOICE,
                             mode: str = "lite") -> str:
    """
    Build a chat prompt that extracts EVERY field present in `document` as one
    flat JSON object of "field": "value" pairs — open-ended, not limited to a
    predefined list. mode="full" appends the verbatim company validation block
    (for the 32B target); mode="lite" appends a compact formatting block.
    """
    rules = (get_invoice_validation_rules() if mode == "full"
             else _EXTRACT_ALL_LITE_RULES)
    user = (
        "Extract ALL details present in the OCR document below as a single flat "
        "JSON object of \"field\": \"value\" pairs. Capture every labelled value you "
        "can find — parties, identifiers, dates, tax and amount lines, bank details, "
        "and anything else actually present in the text. Do NOT invent fields that are "
        "absent.\n"
        f"{rules}\n"
        "Return ONLY the JSON object, with no commentary or markdown fences.\n\n"
        f"Document:\n{document}"
    )
    messages = [
        {"role": "system", "content": EXTRACT_ALL_SYSTEM},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# Kept for parity with the company module / future multi-field JSON demo.
def get_base_extraction_prompt(keyword_json) -> str:
    """BASE PROMPT for structured (JSON) multi-field extraction. Not used by the
    single-field speculative benchmark, but kept so the full company flow is
    available if you later switch the demo to JSON output."""
    base_prompt = f"""You are an expert document information extraction specialist.

================================================================================
TASK: Extract structured data from OCR-processed document text
================================================================================

**TARGET FIELDS (extract these exact keys):**
{json.dumps(keyword_json, indent=4)}

**EXTRACTION INSTRUCTIONS:**
1. Read the OCR-extracted text carefully
2. Extract ONLY the fields specified above
3. Return a valid JSON object with the exact same keys
4. If a value is not found, return "NOT FOUND" for that field

**OUTPUT REQUIREMENTS:**
- Return ONLY the JSON object
- No explanations, markdown, or commentary
- Use double quotes for all JSON strings
"""
    return base_prompt + get_invoice_validation_rules()
