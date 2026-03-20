"""
Dubai Apartment Market Report — PDF
====================================
Usage:  python market_report.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from fpdf import FPDF

ACCENT = (0, 120, 200)
RED = (210, 50, 50)
GREEN = (30, 160, 70)
DARK = (30, 30, 30)
GRAY = (100, 100, 100)
LIGHT_BG = (245, 245, 250)
WHITE = (255, 255, 255)


class Report(FPDF):

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*GRAY)
        self.cell(0, 6, "Dubai Apartment Market Report", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, num: int, title: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*ACCENT)
        self.cell(0, 8, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def para(self, text: str, bold: bool = False):
        self.set_font("Helvetica", "B" if bold else "", 9)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bullet(self, text: str, color=DARK):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*color)
        x = self.get_x()
        self.cell(5, 5, ">")
        self.set_text_color(*DARK)
        self.multi_cell(self.w - x - 15, 5, f" {text}")
        self.ln(0.5)

    def kpi_row(self, items: list[tuple[str, str]]):
        w = (self.w - 20) / len(items)
        self.set_fill_color(*LIGHT_BG)
        y0 = self.get_y()
        for label, value in items:
            x = self.get_x()
            self.set_xy(x, y0)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*GRAY)
            self.cell(w, 4, label, fill=True, new_x="LEFT", new_y="NEXT")
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*DARK)
            self.cell(w, 6, value, fill=True, new_x="LEFT", new_y="NEXT")
            self.set_xy(x + w, y0)
        self.set_y(y0 + 12)
        self.ln(2)

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float], hl: int | None = None):
        self.set_font("Helvetica", "B", 7.5)
        self.set_fill_color(*ACCENT)
        self.set_text_color(*WHITE)
        for i, h in enumerate(headers):
            self.cell(widths[i], 5.5, h, fill=True)
        self.ln()
        for ri, row in enumerate(rows):
            if self.get_y() > 270:
                self.add_page()
                self.set_font("Helvetica", "B", 7.5)
                self.set_fill_color(*ACCENT)
                self.set_text_color(*WHITE)
                for i, h in enumerate(headers):
                    self.cell(widths[i], 5.5, h, fill=True)
                self.ln()
            bg = LIGHT_BG if ri % 2 == 0 else WHITE
            self.set_fill_color(*bg)
            for ci, val in enumerate(row):
                if hl is not None and ci == hl:
                    try:
                        v = float(val.replace("%", "").replace("+", ""))
                        self.set_text_color(*(RED if v < 0 else GREEN))
                        self.set_font("Helvetica", "B", 7.5)
                    except ValueError:
                        self.set_text_color(*DARK)
                        self.set_font("Helvetica", "", 7.5)
                else:
                    self.set_text_color(*DARK)
                    self.set_font("Helvetica", "", 7.5)
                self.cell(widths[ci], 4.5, val, fill=True)
            self.ln()
        self.set_text_color(*DARK)
        self.ln(2)


def build():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 12, "Dubai Apartment Market Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, f"Buyer Opportunity Scanner  |  {date.today().isoformat()}  |  DLD Transactions Feb-Mar 2026", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "Period: 2026-02-02 to 2026-03-19  |  9,923 apartment (flat) transactions", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── 1. Market Structure ─────────────────────────────────────────────
    pdf.section(1, "Market Structure")
    pdf.kpi_row([
        ("Median Price", "AED 1,065,000"),
        ("Mean Price", "AED 1,569,978"),
        ("Mean/Median Ratio", "1.47x"),
        ("Median PSF", "AED 13,190"),
    ])
    pdf.kpi_row([
        ("P10", "AED 528,634"),
        ("P25", "AED 695,000"),
        ("P75", "AED 1,877,128"),
        ("Freehold", "96%"),
    ])
    pdf.para(
        "The market is highly concentrated: the top 5 areas (JVC, Majan, Dubai Marina, "
        "Business Bay, Dubai Creek Harbour) account for 34% of all sales. "
        "Top 10 covers 50%. JVC alone has 1,025 transactions."
    )
    pdf.para(
        "100% of transactions in this dataset are Ready (no off-plan). "
        "96% are Freehold. Non-freehold units sell at a significant discount "
        "(AED 622K median vs AED 1.1M freehold)."
    )

    # ── 2. Price Tiers ──────────────────────────────────────────────────
    pdf.section(2, "Price Tier Segmentation")
    pdf.table(
        ["Tier", "Areas", "Median PSF"],
        [
            ["Ultra-premium", "Bluewaters, Dubai Harbour, Dubai Water Canal", "AED 33K-54K"],
            ["Premium", "Palm Jumeirah, Burj Khalifa, Creek Harbour, Dubai Hills", "AED 23K-25K"],
            ["Mid-market", "Dubai Marina, Business Bay, JLT, Sobha Heartland", "AED 14K-19K"],
            ["Value", "JVC, Arjan, Motor City, Discovery Gardens", "AED 10K-13K"],
            ["Budget", "International City, DLRC, Liwan, Dubai Sports City", "AED 7K-9K"],
        ],
        widths=[28, 90, 30],
    )
    pdf.para("The price gap between ultra-premium and budget is ~8x on PSF (AED 54K Bluewaters vs AED 7K International City).")

    # ── 3. Weekly Price Momentum ────────────────────────────────────────
    pdf.section(3, "Weekly Price Momentum")
    pdf.kpi_row([
        ("Period Median Change", "-5.5%"),
        ("From Peak", "-5.5%"),
        ("Peak Median", "AED 1,140,000"),
        ("Current Median", "AED 1,077,696"),
    ])
    pdf.table(
        ["Week", "Txns", "Txn Chg", "Median", "Med Chg", "Mean", "Skew"],
        [
            ["2026-02-02", "1,441", "N/A",    "AED 1,140,000", "N/A",    "AED 1,789,178", "1.57x"],
            ["2026-02-09", "1,814", "+25.9%",  "AED 1,107,000", "-2.9%",  "AED 1,575,200", "1.42x"],
            ["2026-02-16", "1,529", "-15.7%",  "AED 1,090,000", "-1.5%",  "AED 1,620,078", "1.49x"],
            ["2026-02-23", "1,802", "+17.9%",  "AED 1,100,000", "+0.9%",  "AED 1,531,076", "1.39x"],
            ["2026-03-02", "946",   "-47.5%",  "AED 1,139,419", "+3.6%",  "AED 1,637,218", "1.44x"],
            ["2026-03-09", "1,673", "+76.8%",  "AED 801,000",   "-29.7%", "AED 1,262,779", "1.58x"],
            ["2026-03-16", "718",   "-57.1%",  "AED 1,077,696", "+34.5%", "AED 1,735,013", "1.61x"],
        ],
        widths=[22, 13, 15, 30, 16, 30, 13],
        hl=4,
    )
    pdf.para(
        "The week of March 9 saw a sharp -29.7% median price drop combined with high volume (1,673 txns), "
        "suggesting a wave of lower-priced transactions hitting the market. "
        "The rebound to AED 1.08M the following week came on much lower volume (718 txns), "
        "indicating the recovery is thin."
    )

    # ── 4. Area-Level Opportunities ─────────────────────────────────────
    pdf.section(4, "Buyer Opportunities - Areas With Dropping Prices")
    pdf.para("Areas with declining median prices (first week vs last week, min 50 transactions):", bold=True)
    pdf.table(
        ["Area", "Txns", "First Wk", "Last Wk", "Price Chg", "PSF Chg"],
        [
            ["DUBAI SPORTS CITY",       "306", "AED 670,000",   "AED 520,000",   "-22.4%", "-15.8%"],
            ["ARJAN",                    "345", "AED 891,000",   "AED 704,000",   "-21.0%", "-18.2%"],
            ["BUSINESS BAY",             "591", "AED 1,540,000", "AED 1,345,000", "-12.7%", "-11.5%"],
            ["MAJAN",                    "817", "AED 885,688",   "AED 806,092",   "-9.0%",  "-7.3%"],
            ["SILICON OASIS",            "275", "AED 821,062",   "AED 755,035",   "-8.0%",  "-6.1%"],
            ["JUMEIRAH VILLAGE CIRCLE", "1,025","AED 907,463",   "AED 840,000",   "-7.4%",  "-5.9%"],
            ["DISCOVERY GARDENS",        "267", "AED 601,933",   "AED 560,000",   "-7.0%",  "-5.2%"],
            ["DUBAI MARINA",             "606", "AED 1,820,000", "AED 1,695,000", "-6.9%",  "-4.8%"],
        ],
        widths=[40, 12, 28, 28, 20, 20],
        hl=4,
    )

    # ── 5. 1BR Specific ────────────────────────────────────────────────
    pdf.section(5, "1BR Apartment Opportunities by Area")
    pdf.table(
        ["Area", "1BR Txns", "First Wk", "Last Wk", "Change"],
        [
            ["DUBAI SPORTS CITY",       "134", "AED 664,500",   "AED 520,000",   "-21.7%"],
            ["ARJAN",                    "139", "AED 875,000",   "AED 704,000",   "-19.5%"],
            ["INTERNATIONAL CITY PH 1",  "100", "AED 470,000",   "AED 400,000",   "-14.9%"],
            ["BUSINESS BAY",             "240", "AED 1,400,000", "AED 1,200,000", "-14.3%"],
            ["SILICON OASIS",            "165", "AED 680,000",   "AED 600,000",   "-11.8%"],
            ["MAJAN",                    "395", "AED 673,460",   "AED 610,000",   "-9.4%"],
            ["JUMEIRAH VILLAGE CIRCLE",  "601", "AED 919,825",   "AED 840,000",   "-8.7%"],
            ["DISCOVERY GARDENS",        "113", "AED 755,500",   "AED 700,000",   "-7.3%"],
            ["DUBAI MARINA",             "242", "AED 1,220,165", "AED 1,150,000", "-5.7%"],
        ],
        widths=[40, 18, 30, 30, 22],
        hl=4,
    )

    # ── 6. Areas to Avoid ───────────────────────────────────────────────
    pdf.section(6, "Areas With Rising Prices (Avoid / Monitor)")
    pdf.table(
        ["Area", "Txns", "First Wk", "Last Wk", "Price Chg"],
        [
            ["BURJ KHALIFA",        "349", "AED 2,775,000", "AED 3,480,000", "+25.4%"],
            ["DUBAI CREEK HARBOUR", "365", "AED 2,312,500", "AED 2,600,000", "+12.4%"],
            ["PALM JUMEIRAH",       "204", "AED 3,900,000", "AED 4,100,000", "+5.1%"],
            ["SOBHA HEARTLAND",     "254", "AED 1,349,200", "AED 1,400,000", "+3.8%"],
        ],
        widths=[42, 14, 32, 32, 22],
        hl=4,
    )
    pdf.para(
        "Premium waterfront areas (Burj Khalifa, Creek Harbour, Palm) are moving against the trend, "
        "with prices rising 5-25%. These areas attract international capital and are not "
        "discounting. Avoid for opportunistic buying."
    )

    # ── 7. Key Signals ──────────────────────────────────────────────────
    pdf.section(7, "Key Signals for Cash Buyers")

    pdf.bullet("Volume declining (-30% last 3 wks vs first 3) - fewer buyers competing, stronger negotiating position", RED)
    pdf.bullet("Rising skew (1.57x -> 1.61x) - premium segment decoupling from mid-market; mid-market under more pressure", RED)
    pdf.bullet("Median price falling (-5.5%) - clear downward pressure, good environment for offers below asking", RED)
    pdf.ln(2)

    pdf.para("Top 3 areas with steepest declines:", bold=True)
    pdf.bullet("Dubai Sports City: -22.4% (AED 670K -> AED 520K) - high volume, deep discount", RED)
    pdf.bullet("Arjan: -21.0% (AED 891K -> AED 704K) - significant correction from overheated levels", RED)
    pdf.bullet("Business Bay: -12.7% (AED 1.54M -> AED 1.35M) - prime location, unusual discount window", RED)

    pdf.ln(3)
    pdf.para("Actionable takeaways:", bold=True)
    pdf.bullet("Best value plays: Dubai Sports City and Arjan for sub-AED 750K 1BR units")
    pdf.bullet("Best prime-location play: Business Bay 1BR at AED 1.2M (down from 1.4M)")
    pdf.bullet("Highest-volume liquid market: JVC at AED 840K median, -7.4% and still declining")
    pdf.bullet("Avoid: Burj Khalifa, Creek Harbour, Palm Jumeirah - prices rising against the trend")
    pdf.bullet("Timing: volume is dropping week-over-week, suggesting further softening ahead")

    out = f"report_{date.today().isoformat()}.pdf"
    pdf.output(out)
    return out


if __name__ == "__main__":
    out = build()
    print(f"Report saved to: {out}")
