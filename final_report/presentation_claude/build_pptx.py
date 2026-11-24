"""Editable PowerPoint version of final_report/presentation.html.

    python final_report/presentation_claude/build_pptx.py      (needs python-pptx, lxml, Pillow)

Text, cards, flow boxes, tables and the load / push / snap charts are native PowerPoint
objects (the charts read the saved run files); photos and the matplotlib diagrams are pictures.
"""
import copy
import io
import json
import re
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_LABEL_POSITION
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
FIG = REPORT / "figures"
RUN = REPORT.parent / "Kean" / "results" / "dual_arch_s3_studfit_007"
OUT = HERE / "presentation.pptx"

# ---- theme ---------------------------------------------------------------------------
FONT = "Arial"
THEME = dict(dk1="18212B", lt1="FFFFFF", dk2="5B6876", lt2="E6EEF9", accent1="1F5FAE",
             accent2="2E8B4E", accent3="C0392B", accent4="E0A100", accent5="9ECAE1",
             accent6="FF9F1C", hlink="1F5FAE", folHlink="5B6876")
TC = dict(ink=MSO_THEME_COLOR.TEXT_1, white=MSO_THEME_COLOR.BACKGROUND_1, muted=MSO_THEME_COLOR.TEXT_2,
          soft=MSO_THEME_COLOR.BACKGROUND_2, accent=MSO_THEME_COLOR.ACCENT_1, green=MSO_THEME_COLOR.ACCENT_2,
          red=MSO_THEME_COLOR.ACCENT_3, amber=MSO_THEME_COLOR.ACCENT_4, lightblue=MSO_THEME_COLOR.ACCENT_5,
          orange=MSO_THEME_COLOR.ACCENT_6)
HEX = dict(line="D7DEE6", greensoft="E6F4EA", redsoft="FBE9E7", ambersoft="FDF5DC", code="F1F4F8",
           panel="F6F8FB")

W, H = 13.333, 7.5
LM = 0.6                     # left margin
CW = W - 2 * LM              # content width
TOP = 1.62                   # content top (below title)


def setc(colorformat, key):
    if key in TC:
        colorformat.theme_color = TC[key]
    else:
        colorformat.rgb = RGBColor.from_string(HEX.get(key, key))


def apply_theme(prs):
    part = prs.slide_master.part.part_related_by(RT.THEME)
    root = etree.fromstring(part.blob)
    a = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    root.find(f"{a}themeElements").find(f"{a}clrScheme").set("name", "LEGO Assembly")
    scheme = root.find(f"{a}themeElements/{a}clrScheme")
    for name, val in THEME.items():
        el = scheme.find(a + name)
        for c in list(el):
            el.remove(c)
        etree.SubElement(el, a + "srgbClr", val=val)
    fonts = root.find(f"{a}themeElements/{a}fontScheme")
    fonts.set("name", "LEGO Assembly")
    for k in ("majorFont", "minorFont"):
        fonts.find(f"{a}{k}/{a}latin").set("typeface", FONT)
    part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


# ---- text ----------------------------------------------------------------------------
def runs_of(spec):
    """'plain **bold** plain' or [(text, {bold, color, font, size, italic})] -> list of (text, opts)."""
    if isinstance(spec, list):
        return spec
    out = []
    for i, part in enumerate(re.split(r"\*\*", spec)):
        if part:
            out.append((part, {"bold": True} if i % 2 else {}))
    return out


def fill_tf(tf, paras, size=16, color="ink", bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            bullet=None, start=1, after=6, margin=0.04, line=None):
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, spec in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(after)
        if line:
            p.line_spacing = line
        if bullet:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(Inches(0.3)))
            pPr.set("indent", str(-Inches(0.3)))
            etree.SubElement(pPr, qn("a:buFont"), typeface=FONT)
            if bullet == "num":
                etree.SubElement(pPr, qn("a:buAutoNum"), type="arabicPeriod", startAt=str(start))
            else:
                etree.SubElement(pPr, qn("a:buChar"), char="•")
        for text, o in runs_of(spec):
            r = p.add_run()
            r.text = text
            f = r.font
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            if "font" in o:
                f.name = o["font"]
            setc(f.color, o.get("color", color))
    return tf


def tb(slide, x, y, w, h, paras, name=None, **kw):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    fill_tf(s.text_frame, paras if isinstance(paras, (list, tuple)) and not _is_runs(paras) else [paras], **kw)
    return s


def _is_runs(x):
    return isinstance(x, list) and x and isinstance(x[0], tuple)


def box(slide, x, y, w, h, fill="panel", line="line", radius=0.1, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        line_w=1.0, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(0.5, radius / min(w, h))
    s.shadow.inherit = False
    if fill:
        s.fill.solid()
        setc(s.fill.fore_color, fill)
    else:
        s.fill.background()
    if line:
        setc(s.line.color, line)
        s.line.width = Pt(line_w)
    else:
        s.line.fill.background()
    if name:
        s.name = name
    return s


def card(slide, x, y, w, h, paras, fill="panel", line="line", pad=0.16, size=15, anchor=MSO_ANCHOR.TOP,
         line_w=1.0, name=None, **kw):
    s = box(slide, x, y, w, h, fill=fill, line=line, line_w=line_w, name=name)
    tf = s.text_frame
    fill_tf(tf, paras, size=size, anchor=anchor, margin=pad, **kw)
    tf.margin_top = tf.margin_bottom = Inches(pad * 0.8)
    return s


def why(slide, y, label, text, h=0.62, size=15):
    return card(slide, LM, y, CW, h, [[(label + " ", {"bold": True, "color": "accent"})] + runs_of(text)],
                fill="soft", line=None, size=size, anchor=MSO_ANCHOR.MIDDLE, after=0, name="Callout")


def caption(slide, x, y, w, text, h=0.3, size=11, align=PP_ALIGN.CENTER):
    return tb(slide, x, y, w, h, [text], size=size, color="muted", align=align, after=0, margin=0)


# ---- pictures ------------------------------------------------------------------------
def pic(slide, name, x, y, w, h, cover=False, round_=True, align="center"):
    """Place figures/<name> inside the box: 'contain' keeps it whole, 'cover' crops to fill."""
    iw, ih = Image.open(FIG / name).size
    r, br = iw / ih, w / h
    if cover:
        p = slide.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), Inches(w), Inches(h))
        if r > br:
            c = (1 - br / r) / 2
            p.crop_left = p.crop_right = c
        else:
            c = (1 - r / br) / 2
            p.crop_top = p.crop_bottom = c
    else:
        pw, ph = (w, w / r) if r > br else (h * r, h)
        px = x + (w - pw) / 2 if align == "center" else x
        p = slide.shapes.add_picture(str(FIG / name), Inches(px), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
    if round_:
        geom = p._element.spPr.find(qn("a:prstGeom"))
        geom.set("prst", "roundRect")
        av = geom.find(qn("a:avLst"))
        av.append(etree.fromstring('<a:gd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                                   'name="adj" fmla="val %d"/>' % 2500))
    p.name = name.rsplit(".", 1)[0]
    return p


# ---- charts --------------------------------------------------------------------------
def style_chart(chart, legend=False):
    chart.font.size = Pt(10)
    chart.font.name = FONT
    setc(chart.font.color, "muted")
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(10)
    for ax in (chart.category_axis, chart.value_axis):
        ax.format.line.color.rgb = RGBColor.from_string(HEX["line"])
        ax.tick_labels.font.size = Pt(10)
    chart.value_axis.has_major_gridlines = True
    gl = chart.value_axis.major_gridlines.format.line
    gl.color.rgb = RGBColor.from_string(HEX["line"])
    gl.width = Pt(0.75)
    chart.value_axis.format.line.fill.background()


def chart_title(chart, text):
    chart.has_title = True
    tf = chart.chart_title.text_frame
    tf.text = text
    f = tf.paragraphs[0].runs[0].font
    f.size, f.bold, f.name = Pt(12), True, FONT
    setc(f.color, "ink")


def axis_title(ax, text):
    ax.has_title = True
    tf = ax.axis_title.text_frame
    tf.text = text
    f = tf.paragraphs[0].runs[0].font
    f.size, f.bold, f.name = Pt(10), False, FONT
    setc(f.color, "muted")


def series_to_line(chart, idx, color, dash=False, marker=None, line=True, width=1.25):
    """Move bar series `idx` onto a line chart sharing the bar chart's axes (a native combo chart)."""
    pa = chart._chartSpace.find(qn("c:chart")).find(qn("c:plotArea"))
    bar = pa.find(qn("c:barChart"))
    ser = bar.findall(qn("c:ser"))[idx]
    bar.remove(ser)
    for tag in ("c:invertIfNegative", "c:spPr", "c:dPt"):
        for el in ser.findall(qn(tag)):
            ser.remove(el)
    A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
    C = 'xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"'
    ln = ('<a:ln %s w="%d"><a:solidFill><a:srgbClr val="%s"/></a:solidFill>%s</a:ln>'
          % (A, Pt(width), color, '<a:prstDash val="dash"/>' if dash else "") if line
          else '<a:ln %s><a:noFill/></a:ln>' % A)
    sppr = etree.fromstring('<c:spPr %s %s>%s</c:spPr>' % (C, A, ln))
    if marker:
        mk = etree.fromstring('<c:marker %s %s><c:symbol val="%s"/><c:size val="%d"/><c:spPr>'
                              '<a:solidFill><a:srgbClr val="%s"/></a:solidFill><a:ln><a:solidFill>'
                              '<a:srgbClr val="%s"/></a:solidFill></a:ln></c:spPr></c:marker>'
                              % (C, A, marker[0], marker[1], color, color))
    else:
        mk = etree.fromstring('<c:marker %s><c:symbol val="none"/></c:marker>' % C)
    tx = ser.find(qn("c:tx"))
    tx.addnext(sppr)
    sppr.addnext(mk)
    etree.SubElement(ser, qn("c:smooth"), val="0")
    lc = etree.fromstring('<c:lineChart %s><c:grouping val="standard"/><c:varyColors val="0"/></c:lineChart>' % C)
    lc.append(ser)
    etree.SubElement(lc, qn("c:marker"), val="1")
    for ax in bar.findall(qn("c:axId")):
        lc.append(copy.deepcopy(ax))
    bar.addnext(lc)


def color_points(series, keys):
    for i, k in enumerate(keys):
        pt = series.points[i]
        pt.format.fill.solid()
        setc(pt.format.fill.fore_color, k)


def legend_row(slide, x, y, items, size=11):
    """A hand-made legend for per-bar colours (PowerPoint's legend shows one colour per series)."""
    for key, label, kind in items:
        if kind == "line":
            s = slide.shapes.add_connector(1, Inches(x), Inches(y + 0.1), Inches(x + 0.3), Inches(y + 0.1))
            s.line.color.rgb = RGBColor.from_string("18212B")
            s.line.dash_style = 4
            s.line.width = Pt(1.25)
        else:
            box(slide, x + 0.05, y + 0.02, 0.18, 0.16, fill=key, line=None, shape=MSO_SHAPE.RECTANGLE)
        t = tb(slide, x + 0.36, y - 0.04, 4, 0.28, [label], size=size, color="muted", after=0, margin=0)
        t.text_frame.word_wrap = False
        x += 0.7 + len(label) * size * 0.0068


# ---- layouts -------------------------------------------------------------------------
def setup_layouts(prs):
    master = prs.slide_master
    title_style = master._element.find(qn("p:txStyles")).find(qn("p:titleStyle")).find(qn("a:lvl1pPr"))
    title_style.set("algn", "l")
    d = title_style.find(qn("a:defRPr"))
    d.set("sz", "3000")
    d.set("b", "1")
    title_only = prs.slide_layouts[5]
    t = title_only.placeholders[0]
    t.left, t.top, t.width, t.height = Inches(LM), Inches(0.72), Inches(CW), Inches(0.72)
    bp = t._element.txBody.find(qn("a:bodyPr"))
    bp.set("anchor", "t")
    bp.set("lIns", "0")
    bp.set("rIns", "0")
    # drop the stock date / footer / number placeholders; add our own footer and page number
    for ph in list(title_only.placeholders)[1:]:
        ph._element.getparent().remove(ph._element)
    tmp = prs.slides.add_slide(prs.slide_layouts[6])
    f = tb(tmp, LM, 7.02, 7, 0.3, ["Dual-Arm Robotic LEGO Assembly · Group 10"], size=10, color="muted",
           after=0, margin=0, name="Footer")
    n = tb(tmp, W - LM - 2, 7.02, 2, 0.3, [""], size=10, color="muted", after=0, margin=0,
           align=PP_ALIGN.RIGHT, name="Slide number")
    p = n.text_frame.paragraphs[0]._p
    for r in p.findall(qn("a:r")):
        p.remove(r)
    rpr = '<a:rPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" lang="en-US" sz="1000">' \
          '<a:solidFill><a:schemeClr val="tx2"/></a:solidFill></a:rPr>'
    fld = etree.SubElement(p, qn("a:fld"), id="{5C1D2E3F-1111-4A2B-9C3D-0123456789AB}", type="slidenum")
    fld.append(etree.fromstring(rpr))
    etree.SubElement(fld, qn("a:t")).text = "‹#›"
    tail = etree.SubElement(p, qn("a:r"))
    tail.append(etree.fromstring(rpr))
    etree.SubElement(tail, qn("a:t")).text = " / 24"
    end = p.find(qn("a:endParaRPr"))
    if end is not None:
        p.remove(end)
        p.append(end)
    tree = title_only.shapes._spTree
    for s in (f, n):
        tree.append(s._element)
    # remove the temporary slide
    sld = prs.slides._sldIdLst[-1]
    prs.part.drop_rel(sld.rId)
    prs.slides._sldIdLst.remove(sld)
    title_only.name = "Title Only + footer"

    ts = prs.slide_layouts[0]
    t, sub = ts.placeholders[0], ts.placeholders[1]
    t.left, t.top, t.width, t.height = Inches(LM + 0.3), Inches(2.25), Inches(CW - 0.6), Inches(1.6)
    sub.left, sub.top, sub.width, sub.height = Inches(LM + 0.3), Inches(4.0), Inches(CW - 1.2), Inches(1.6)
    for ph in (t, sub):
        bp = ph._element.txBody.find(qn("a:bodyPr"))
        bp.set("anchor", "t")
        bp.set("lIns", "0")
    for ph in list(ts.placeholders)[2:]:
        ph._element.getparent().remove(ph._element)


def content(prs, kicker, title):
    s = prs.slides.add_slide(prs.slide_layouts[5])
    tb(s, LM, 0.42, CW, 0.3, [kicker.upper()], size=12, bold=True, color="accent", after=0, margin=0,
       name="Kicker")
    s.shapes.title.text = title
    return s


def title_slide(prs, kicker, title, lead, size=40):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    bg = s.background.fill
    bg.solid()
    setc(bg.fore_color, "F4F7FC")
    tb(s, LM + 0.3, 1.8, CW, 0.35, [kicker.upper()], size=13, bold=True, color="accent", after=0, margin=0,
       name="Kicker")
    tf = s.shapes.title.text_frame
    tf.text = title[0]
    for line in title[1:]:
        tf.add_paragraph().text = line
    for p in tf.paragraphs:
        p.alignment = PP_ALIGN.LEFT
        for r in p.runs:
            r.font.size, r.font.bold = Pt(size), True
            setc(r.font.color, "ink")
    fill_tf(s.placeholders[1].text_frame, [lead], size=18, color="muted", margin=0, line=1.15)
    return s


# ---- the deck ------------------------------------------------------------------------
def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    apply_theme(prs)
    setup_layouts(prs)
    plans = {a: json.loads((RUN / "plans" / ("%s.json" % a)).read_text()) for a in "AB"}

    # 1 ---------------------------------------------------------------------------------
    s = title_slide(prs, "ISDN Robotic Fabrication with Python · Group 10",
                    ["Two robot arms that build LEGO", "and hold each other's work steady"],
                    "How the system works, from a drawing to two finished structures, shown on the final run: "
                    "arm A builds an arch from a drawing, arm B builds the S3 bridge from a brick list, and "
                    "every brick is pushed onto studs that resist.")
    x = LM + 0.3
    for chip in ("Newton physics (GPU)", "cuRobo motion planning", "force-balance linear program",
                 "2 × Franka FR3"):
        w = 0.4 + len(chip) * 0.092
        card(s, x, 5.55, w, 0.42, [chip], fill="white", line="line", size=13, pad=0.05, color="muted",
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, after=0, name="Chip")
        s.shapes[-1].adjustments[0] = 0.5
        x += w + 0.15

    # 2 ---------------------------------------------------------------------------------
    s = content(prs, "The task", "Three requirements, all met in the final run")
    reqs = [("Take the target structure directly as input.", "The arch is a drawing; S3 is a brick list."),
            ("Work out the build order and where support is needed, automatically.",
             "Order, grips, press points and supports are all generated."),
            ([("Both arms build two different structures", {"bold": True}),
              (", ≥ 8 bricks each, each with a part needing support.", {})],
             "Arch: 11 bricks, 1 support. S3: 16 bricks, 3 supports.")]
    for i, (head, sub) in enumerate(reqs):
        y = TOP + 0.25 + i * 1.45
        c = box(s, LM, y, 0.5, 0.5, fill="green", line=None, shape=MSO_SHAPE.OVAL, name="Requirement number")
        fill_tf(c.text_frame, [str(i + 1)], size=18, bold=True, color="white", align=PP_ALIGN.CENTER,
                anchor=MSO_ANCHOR.MIDDLE, after=0, margin=0)
        head = head if isinstance(head, list) else [(head, {"bold": True})]
        tb(s, LM + 0.75, y - 0.04, 6.3, 1.25, [head, [(sub, {"color": "muted"})]], size=17, after=4)
    pic(s, "fin_done_s334.jpg", 7.9, TOP + 0.2, 4.83, 3.6, cover=True)
    caption(s, 7.9, TOP + 3.9, 4.83, "End of the final run: S3 (front) and the arch (back)", size=12)

    # 3 ---------------------------------------------------------------------------------
    s = content(prs, "The final run at a glance", "")
    s.shapes.title.text_frame.paragraphs[0].text = ""
    p = s.shapes.title.text_frame.paragraphs[0]
    p.add_run().text = "Run "
    r = p.add_run()
    r.text = "dual_arch_s3_studfit_007"
    r.font.name = "Courier New"
    stats = [("27/27", "bricks placed: arch 11, S3 16"),
             ("4/4", "supports held by the other arm (arch 10; S3 13, 14, 15)"),
             ("1,907", "frames with both arms building at the same time"),
             ("199 / 0", "arm motions planned / failed"),
             ("1.46 mm", "worst sideways error of any brick (limit 2 mm)"),
             ("0", "unintended contacts or broken joints"),
             ("515 s", "simulated time for both structures"),
             ("PASS", "independent check of the saved run")]
    gw = (CW - 3 * 0.25) / 4
    for i, (num, lab) in enumerate(stats):
        x, y = LM + (i % 4) * (gw + 0.25), TOP + 0.1 + (i // 4) * 1.55
        last = num == "PASS"
        card(s, x, y, gw, 1.35, [[(num, {"size": 32, "bold": True, "color": "green" if last else "accent"})],
                                 [(lab, {"color": "muted"})]], fill="greensoft" if last else "white",
             line="green" if last else "line", size=13, after=2, name="Stat")
    fy, fh = 5.05, 1.5
    flow = [("Input", "drawing or brick list", "greensoft", "green"),
            ("1. Blueprint", "which bricks, where", "soft", "accent"),
            ("2. Plan", "order, grip, press point, where to support", "soft", "accent"),
            ("3. Control", "motion, pressing, holding, collision checks", "soft", "accent"),
            ("4. Teamwork & completion", "two arms, one world", "redsoft", "red")]
    aw = 0.42
    bw = (CW - 4 * aw) / 5
    for i, (head, sub, fill, line) in enumerate(flow):
        x = LM + i * (bw + aw)
        card(s, x, fy, bw, fh, [[(head, {"bold": True, "size": 16})], sub], fill=fill, line=line, line_w=2,
             size=13, after=3, pad=0.12, name="Flow step")
        if i < 4:
            a = box(s, x + bw + 0.08, fy + fh / 2 - 0.15, aw - 0.16, 0.3, fill="accent", line=None,
                    shape=MSO_SHAPE.RIGHT_ARROW, name="Flow arrow")

    # 4 ---------------------------------------------------------------------------------
    s = content(prs, "The simulated world", "A physics simulation of the work cell")
    tb(s, LM, TOP + 0.1, 4.5, 4.8,
       ["**2 × Franka FR3** arms, 0.9 m apart; A on the left, B on the right.",
        "Each arm has its own **studded base plate** and its own row of **feeder bricks**.",
        "**Newton 1.2.1 + MuJoCo solver** on the GPU: 60 frames per second, each split into 16 small steps.",
        "Every step: find what touches what, work out the forces, move every body."],
       size=17, bullet="dot", after=12)
    iw = (CW - 4.9 - 0.25) / 2
    for i, (img, cap) in enumerate((("fin_overview_top.jpg", "From above: A builds the arch, B builds S3"),
                                    ("fin_overview_34.jpg", "Both arms at work (t = 150 s)"))):
        x = LM + 4.9 + i * (iw + 0.25)
        pic(s, img, x, TOP + 0.35, iw, iw / 1.6)
        caption(s, x, TOP + 0.45 + iw / 1.6, iw, cap, size=12)

    # 5 ---------------------------------------------------------------------------------
    s = content(prs, "The simulated world · Bricks", "How a brick is modelled, and how it goes onto the studs")
    pic(s, "diagram_studfit.png", LM, TOP, CW, 2.35, round_=False)
    half = (CW - 0.4) / 2
    tb(s, LM, 4.2, half, 2.6,
       [[("Stud fit", {"bold": True})]], size=17, after=4)
    tb(s, LM, 4.6, half, 2.3,
       ["Brick reaches the stud tops → **held there**",
        "Push reaches **8.9 N per stud** (4 studs: 35.6 N; 8 studs: 71.2 N)",
        "Brick **slides straight down** and seats",
        "**Snap check** passes → locked in"], size=16, bullet="num", after=6)
    x2 = LM + half + 0.4
    pic(s, "diagram_brick.png", x2, 4.2, half, 1.15, round_=False)
    tb(s, x2, 5.5, half, 1.3,
       ["You see the full brick; contacts use simple hidden shapes: a top slab, four walls, one cylinder per "
        "stud. Fast and stable."], size=16)

    # 6 ---------------------------------------------------------------------------------
    s = content(prs, "The simulated world · Holding and breaking", "Locking bricks together, and breaking")
    cards = [("Welds = the clutch", "Once a brick passes the snap check, invisible welds switch on between it and "
              "each brick it sits on. It then behaves like a clicked-in brick.", "line", "white", "accent"),
             ("Joints can break", "Every frame each weld's load becomes a **utilisation u** (load ÷ strength). "
              "If **u ≥ 1 twice in a row**, the weld is deleted and the run fails.", "line", "white", "red"),
             ("Same rule as the planner", "The planner predicts u with the same formula, so plan and physics agree "
              "on what “too much load” means.", "line", "white", "accent"),
             ("During a supported press", "The break check pauses until the helper lets go: welds report a short "
              "load spike as they lock on, and the holding hand is there to absorb it.", "amber", "ambersoft",
              "amber")]
    cw2, ch2 = (CW - 0.3) / 2, 2.35
    for i, (head, body, line, fill, dot) in enumerate(cards):
        x, y = LM + (i % 2) * (cw2 + 0.3), TOP + 0.1 + (i // 2) * (ch2 + 0.3)
        card(s, x, y, cw2, ch2, [[(head, {"bold": True, "size": 19})], body], fill=fill, line=line,
             line_w=1.5 if line == "amber" else 1, size=17, after=8, pad=0.25, name="Card")

    # 7 ---------------------------------------------------------------------------------
    s = content(prs, "Stage 1 · Drawing → blueprint", "Carve with shadows, then tile with bricks")
    pic(s, "input_arch_front.png", LM, TOP, 4.4, 1.85, round_=False)
    caption(s, LM, TOP + 1.9, 4.4, "Input: the arch drawing (front view, 10 studs wide, 2 deep)", size=12)
    tb(s, 5.3, TOP - 0.05, CW - 4.7, 2.5,
       ["**Find the shape**: threshold halfway between darkest and brightest; crop.",
        "**Fit to the LEGO grid**: 1 stud = 8.0 mm, 1 layer = 9.6 mm.",
        "**Majority vote**: a grid square is full if more than half is dark.",
        "**Carve**: keep a cube only if every view sees it (“visual hull”).",
        "**Tile each layer** with bricks, bottom up, overhangs first.",
        "**Number** the bricks in build order."], size=15, bullet="num", after=3)
    pic(s, "pipeline_arch.png", LM, 4.15, CW, 2.75, round_=False)

    # 8 ---------------------------------------------------------------------------------
    s = content(prs, "Stage 1 · Drawing → blueprint", "How each brick is chosen")
    tb(s, LM, TOP, CW, 0.4, ["Every brick that could cover the next cube is scored, in this order:"], size=17)
    tb(s, LM, TOP + 0.5, half, 2.0,
       ["rests on the **most different bricks below** (ties the arch's legs together)",
        "**fewest joints on top of joints** (like a brick wall)",
        "**bigger** bricks"], size=17, bullet="num", after=6)
    tb(s, x2, TOP + 0.5, half, 1.0,
       ["runs **across** the layer below", "closer to the centre"], size=17, bullet="num", start=4, after=6)
    card(s, x2, TOP + 1.45, half, 0.85,
         [[("A brick with nothing below it → “overhangs too far”, never an unbuildable blueprint.",
            {"color": "muted"})]], fill="panel", line=None, size=15, anchor=MSO_ANCHOR.MIDDLE, after=0)
    pic(s, "layers_arch.png", LM, 4.25, CW, 1.6, round_=False)
    caption(s, LM, 6.05, CW, "The arch, layer by layer (numbers = build step): the 2×6 crown, step 10, spans "
            "the gap", size=12)

    # 9 ---------------------------------------------------------------------------------
    s = content(prs, "Stage 2 · Blueprint → plan", "One step per brick")
    cards = [("When", "bottom layer first; a brick only after everything under it. 2×2s first within a layer "
              "(16 mm faces vs 17.5 mm finger pads)."),
             ("Where", "target pose from the grid position, plus a **hover point 25 mm above** where the arm stops "
              "before going down."),
             ("How to hold", "the pair of faces whose finger pads touch the fewest neighbours; fingers open to brick "
              "width − 1.2 mm."),
             ("Where to press", "on an overhang, slide the press point along the brick to the spot with the "
              "**least strain** on the structure.")]
    ch3 = 1.7
    for i, (head, body) in enumerate(cards):
        x, y = LM + (i % 2) * (cw2 + 0.3), TOP + 0.1 + (i // 2) * (ch3 + 0.3)
        card(s, x, y, cw2, ch3, [[(head, {"bold": True, "size": 19, "color": "accent"})], body], fill="white",
             size=17, after=6, pad=0.22, name="Card")
    why(s, 5.85, "Why?", "Layer by layer can never place a brick in mid-air, and grips and press points use the "
        "same force model as the support decision.", h=0.8, size=16)

    # 10 --------------------------------------------------------------------------------
    s = content(prs, "Stage 2b · Where the other arm must hold", "Will pressing this brick break the structure?")
    pic(s, "diagram_lever.png", LM, TOP, half, 2.95, round_=False)
    tb(s, LM, 4.75, half, 1.6,
       ["A press on an overhang acts like standing at the end of a diving board: it pulls the studs at the root "
        "apart. Studs hold **11.3 N each**."], size=16)
    tb(s, x2, TOP, half, 5.2,
       ["**Utilisation u** = prying load ÷ what the studs can hold. u ≥ 1 breaks.",
        "A **linear program** balances forces and moments on every brick (weight, press, helper's grip) and finds "
        "the load sharing with the **lowest worst u**.",
        "Press = n × 8.9 N, checked with a 3 N sideways push in 4 directions.",
        "**Needs help** if u × 3 ≥ 1 (safety factor 3), or if the brick overhangs.",
        "**Grip choice**: must hold a brick on the press's **load path** (the column under it); among safe grips, "
        "the one **nearest the press**."], size=16, bullet="dot", after=9)

    # 11 -- native load chart -----------------------------------------------------------
    s = content(prs, "Stage 2b · Where the other arm must hold", "The final run's plans")
    total = 11 + 16
    plot_w = CW - 0.2
    x = LM
    for arm, ttl in (("A", "Arch (arm A)"), ("B", "S3 (arm B)")):
        seq = plans[arm]["sequence"]
        u0 = [round(st["predicted_util_unbraced"] or 0, 4) for st in seq]
        ub = [round(st["brace"]["predicted_util_braced"], 4) if st["requires_brace"] else None for st in seq]
        cd = CategoryChartData()
        cd.categories = [str(st["step"]) for st in seq]
        cd.add_series("without help", u0)
        cd.add_series("with the planned support", ub)
        cd.add_series("limit for building alone (u × 3 ≥ 1)", [round(1 / 3, 4)] * len(seq))
        w = plot_w * len(seq) / total + (0.25 if arm == "A" else 0)
        gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(TOP + 0.05), Inches(w),
                                Inches(4.05), cd)
        gf.name = "Load chart " + arm
        ch = gf.chart
        style_chart(ch)
        chart_title(ch, ttl)
        axis_title(ch.category_axis, "build step")
        if arm == "A":
            axis_title(ch.value_axis, "predicted joint load u")
        ch.value_axis.maximum_scale = 1.0 if arm == "A" else 3.6
        ch.value_axis.minimum_scale = 0
        ch.value_axis.tick_labels.number_format = "0.0"
        ch.value_axis.tick_labels.number_format_is_linked = False
        plot = ch.plots[0]
        plot.gap_width, plot.overlap = 60, 0
        color_points(plot.series[0], ["red" if st["requires_brace"] else "orange" if u * 3 >= 1 else "lightblue"
                                      for st, u in zip(seq, u0)])
        plot.series[1].format.fill.solid()
        setc(plot.series[1].format.fill.fore_color, "green")
        series_to_line(ch, 2, "18212B", dash=True)
        x += w + 0.2 - (0.0 if arm == "A" else 0)
    legend_row(s, LM + 0.2, 5.82, [("lightblue", "safe alone", "box"),
                                   ("red", "gets a support", "box"),
                                   ("orange", "over limit, no grip helps", "box"),
                                   ("green", "with the support", "box"),
                                   ("ink", "limit for building alone (u \u00d7 3 = 1)", "line")])
    caption(s, LM, 6.22, CW, "Orange: S3 step 11 is over the limit, but every reachable grip is on the other pier "
            "(off the load path), so it is built alone. Step 15 is supported because it overhangs.", h=0.5,
            size=12, align=PP_ALIGN.LEFT)

    # 12 --------------------------------------------------------------------------------
    s = content(prs, "Stage 2b · Where the other arm must hold", "The four supported steps")
    pic(s, "final_supports.png", LM, TOP + 0.15, CW, 3.6, round_=False)
    for i, (lab, before, after) in enumerate((("Arch 10", "0.95", "0.95"), ("S3 13", "3.43", "0.12"),
                                              ("S3 14", "1.96", "0"), ("S3 15", "0.06", "0"))):
        x = LM + i * (CW + 0.25) / 4
        card(s, x, 5.15, (CW - 0.75) / 4, 0.75,
             [[(lab + "   ", {"bold": True}), (before, {"color": "red", "bold": True}),
               (" → ", {"color": "muted"}), (after, {"color": "green", "bold": True})]],
             fill="white", size=16, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, after=0, name="Load change")
    caption(s, LM, 6.15, CW, "Hatched: the brick being pressed. Dashed: planned grip. Filled: the grip the helper "
            "actually used, a nearby alternative its search found. Numbers: predicted load u without → with "
            "the support.", h=0.5, size=12, align=PP_ALIGN.LEFT)

    # 13 --------------------------------------------------------------------------------
    s = content(prs, "Stage 3 · Plan → motion", "Plan the big moves, script the small ones")
    for i, (img, cap) in enumerate((("fin_pick.jpg", "to feeder — planned path"),
                                    ("fin_carry.jpg", "transport — planned path, brick attached as 12 spheres"))):
        x = LM + i * (cw2 + 0.3)
        pic(s, img, x, TOP, cw2, 3.1, cover=True)
        caption(s, x, TOP + 3.15, cw2, cap, size=12)
    card(s, LM, 5.25, cw2, 1.45, ["**cuRobo** plans long moves on the GPU: arm = spheres, world = boxes (plates, "
                                  "structures, feeders, the other arm). **Last 30 mm** = straight line, checked every "
                                  "2 mm."], fill="white", size=16, pad=0.22, name="Card")
    card(s, LM + cw2 + 0.3, 5.25, cw2, 1.45, ["**Inverse kinematics** turns the hand target into 7 joint angles "
                                              "every frame. The controller **steers the brick**, not the hand, by "
                                              "up to 4 mm."], fill="white", size=16, pad=0.22, name="Card")

    # 14 --------------------------------------------------------------------------------
    s = content(prs, "Manipulation", "Pressing each brick onto the studs")
    tw = (CW - 0.5) / 3
    for i, (img, cap) in enumerate((("fin_on_studs.jpg", "1. lowered onto the stud tops"),
                                    ("fin_press.jpg", "2. released, fist closed, pressed"),
                                    ("fin_seated.jpg", "3. seated; the hand lifts away"))):
        x = LM + i * (tw + 0.25)
        pic(s, img, x, TOP, tw, tw / 1.6)
        caption(s, x, TOP + tw / 1.6 + 0.07, tw, cap, size=12)
    tb(s, LM, 4.75, CW, 2.0,
       ["The side grip is gentle (~1.5 N per finger) and would slip, so the arm **presses from above with closed "
        "fingertips**.",
        "Force ramps over 0.5 s to **1.3× what the studs need**; joints 5× stiffer so the hand does not "
        "skid; snap check 0.25 s after it goes in."], size=17, bullet="dot", after=10)

    # 15 -- native push chart -----------------------------------------------------------
    s = content(prs, "Manipulation", "The push on every brick")
    rows = [json.loads(line) for line in (RUN / "episodes.jsonl").read_text().splitlines()]
    supported = {st["brick_id"] for pl in plans.values() for st in pl["sequence"] if st["requires_brace"]}
    push = {e["brick"]: e for e in json.loads((RUN / "stud_fit.json").read_text()) if e["event"] == "pushed_through"}
    order = [e["brick_id"] for e in rows]
    cd = CategoryChartData()
    cd.categories = [("A" if b.startswith("A") else "B") + (b[-2:].lstrip("0") or "0") for b in order]
    cd.add_series("push when the brick went in", [round(push[b]["peak_N"], 1) for b in order])
    cd.add_series("force its studs need (n × 8.9 N)", [round(push[b]["needed_N"], 1) for b in order])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(LM), Inches(TOP), Inches(CW), Inches(3.75), cd)
    gf.name = "Push chart"
    ch = gf.chart
    style_chart(ch)
    axis_title(ch.value_axis, "force (N)")
    axis_title(ch.category_axis, "bricks in the order they were seated (A = arch, B = S3)")
    ch.value_axis.minimum_scale, ch.value_axis.maximum_scale, ch.value_axis.major_unit = 0, 80, 20
    ch.plots[0].gap_width = 50
    color_points(ch.plots[0].series[0], ["green" if b in supported else "lightblue" for b in order])
    series_to_line(ch, 1, "18212B", marker=("dash", 14), line=False)
    legend_row(s, LM + 0.2, 5.45, [("lightblue", "unsupported brick: push", "box"),
                                   ("green", "supported brick: push", "box"),
                                   ("ink", "force the studs need (black dash)", "box")])
    why(s, 5.9, "Supported press:", "ramps over 2 s to 0.9× the needed force, and the brick is let through at "
        "**0.85×** (green). The holding hand carries part of the load; a full-strength press on a held structure "
        "can break the two-stud joint under S3's last brick.", h=0.95, size=15)

    # 16 --------------------------------------------------------------------------------
    s = content(prs, "Manipulation · The helper", "Holding the other arm's structure")
    for i, (img, cap) in enumerate((("sim_sf_crown_low.jpg", "Arch crown: A presses, B holds"),
                                    ("sim_sf_s3_13_side.jpg", "S3 step 13: A holds, B presses"),
                                    ("sim_sf_s3_15_side.jpg", "S3 step 15: A holds, B presses"))):
        x = LM + i * (tw + 0.25)
        pic(s, img, x, TOP, tw, 2.8, cover=True)
        caption(s, x, TOP + 2.85, tw, cap, size=12)
    card(s, LM, 4.95, cw2, 1.85, ["**Search → check → guarded approach → close → hold.** Up to 8 "
                                 "poses; checks its own path and the builder's; last 10 mm at 5 mm/s, stops on touch; "
                                 "both fingers must feel force; holds at 20× stiffness (37–40 N grip)."],
         fill="white", size=15, pad=0.2, name="Card")
    card(s, LM + cw2 + 0.3, 4.95, cw2, 1.85, ["**15 attempts, 4 held.** Rejected: 3 fingers too close, 1 blocking "
                                             "the brick's path, 3 no safe path. 4 guard stops: lift clear, back out "
                                             "along the hand axis, try a column ≥ 12 mm away."],
         fill="white", size=15, pad=0.2, name="Card")

    # 17 --------------------------------------------------------------------------------
    s = content(prs, "Collision avoidance", "Spheres, boxes and five layers of protection")
    layers = [("1 · Plan", "cuRobo avoids every box: plates, structures, feeders, the other arm."),
              ("2 · Re-check", "every path re-checked in small steps with the full sphere model of both arms, "
                                    "real finger openings and both carried bricks."),
              ("3 · Book", "reserve the space a move will sweep; overlapping bookings wait."),
              ("4 · Guard", "every command, every frame: ≥ 1 mm from anything not meant to be touched."),
              ("5 · Audit", "every physics contact classified; any unintended one fails the run.")]
    for i, (head, body) in enumerate(layers):
        x = LM + i * 0.3
        card(s, x, TOP + i * 1.0, 7.2 - i * 0.3, 0.88, [[(head, {"bold": True, "color": "accent"}),
                                                         (" — " + body, {})]],
             fill="soft", line=None, size=14, anchor=MSO_ANCHOR.MIDDLE, after=0, pad=0.15, name="Protection layer")
    rx = LM + 7.55
    pic(s, "diagram_spheres.png", rx, TOP, CW - 7.55, 2.2, round_=False)
    card(s, rx, TOP + 2.45, CW - 7.55, 2.4, ["**Final run:** 199 motions planned, 0 failed; contacts only intended "
                                             "(32 bricks rubbing a neighbour while seating, 24 stud matings)."],
         fill="greensoft", line="green", size=16, pad=0.22, name="Card")

    # 18 --------------------------------------------------------------------------------
    s = content(prs, "Collision avoidance · Two arms", "Sharing space, and working next to a helper")
    card(s, LM, TOP + 0.1, cw2, 3.75,
         [[("Booking space", {"bold": True, "size": 19})],
          "Each arm plans on its own. Before moving it books the space its spheres will sweep; a booking must stay "
          "≥ 25 mm from the other arm's. Straight moves book their space plus 1.5× the hand's travel, and "
          "stop if they leave it. No combined 14-joint planner needed."], fill="white", size=17, after=8, pad=0.25,
         name="Card")
    card(s, LM + cw2 + 0.3, TOP + 0.1, cw2, 3.75,
         [[("Next to a holding helper", {"bold": True, "size": 19})],
          "A support only works close up, so small overlaps are allowed there: fingers ≤ 1.25 mm into a brick's "
          "box, the builder's path ≤ 2 mm into the helper's spheres; bricks may move ≤ 2 mm while pressed. "
          "Smallest gap in the run: −0.90 mm, inside these allowances."], fill="ambersoft", line="amber",
         line_w=1.5, size=17, after=8, pad=0.25, name="Card")
    why(s, 5.85, "Why?", "The same sphere model in planner and checker means they agree on geometry; the physics "
        "contact audit is the final, independent check.", h=0.8, size=16)

    # 19 --------------------------------------------------------------------------------
    s = content(prs, "Collaboration between the arms", "The coordinator: build, ask, help")
    tb(s, LM, TOP - 0.05, half, 1.45,
       ["Each arm follows its own plan; both build at once.",
        "Help is asked **before pickup**, granted when **both arms are free**.",
        "Both ask at once → **A first**, B's request kept and served next."], size=15, bullet="dot", after=4)
    tb(s, x2, TOP - 0.05, half, 1.3,
       ["**Don't run ahead**: an arm ahead waits if the other's next step needs help.",
        "A finished builder stays on as helper.",
        "“Both hold a brick and wait” cannot happen."], size=15, bullet="dot", after=4)
    pic(s, "final_timeline.png", LM, 3.15, CW, 3.2, round_=False)
    caption(s, LM, 6.45, CW, "258 s: B waits (A's next step needs help) · 265 s: A asks; B holds the arch for "
            "the crown · 350–516 s: A, finished, holds S3 for steps 13–15", size=12)

    # 20 -- table + native snap charts --------------------------------------------------
    s = content(prs, "Task completion", "When is a brick placed? When is the job done?")
    lw = 4.5
    tbl = s.shapes.add_table(4, 2, Inches(LM), Inches(TOP), Inches(lw), Inches(1.6)).table
    for r, (a, b) in enumerate((("Snap check", "limit"), ("sideways", "2 mm"), ("height", "1.5 mm"),
                                ("rotation / tilt", "5° / 5°"))):
        for c, v in enumerate((a, b)):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            setc(cell.fill.fore_color, "soft" if r == 0 else "white")
            fill_tf(cell.text_frame, [v], size=15, bold=r == 0, color="muted" if r == 0 else "ink", after=0,
                    margin=0.1)
    tbl.columns[0].width, tbl.columns[1].width = Inches(2.7), Inches(lw - 2.7)
    tbl.first_row = True
    tb(s, LM, TOP + 1.8, lw, 1.1, ["Pass → welds on, step complete. Fail → run stops. A brick left on the "
                                   "stud tops (1.7 mm high) always fails."], size=15)
    tb(s, LM, TOP + 2.85, lw, 2.4,
       [[("Independent check", {"bold": True}),
         (" reads only the saved files: every step mated, every support asked for, held with two-finger force and "
          "released in order, both arms really moved together, no bad contacts or breaks. ", {}),
         ("Final run: pass.", {"bold": True, "color": "green"})]], size=15)
    rx, rw = LM + lw + 0.35, CW - lw - 0.35
    sw = rw * 0.55
    xy = XyChartData()
    for arm, lab in (("A", "arch (arm A)"), ("B", "S3 (arm B)")):
        ser = xy.add_series(lab)
        for e in rows:
            if e["structure"].startswith(arm):
                ser.add_data_point(round(e["final_error_mm"], 3), round(e["dz_mm"], 3))
    gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(rx), Inches(TOP), Inches(sw), Inches(4.6), xy)
    gf.name = "Snap scatter"
    ch = gf.chart
    style_chart(ch, legend=True)
    ch.category_axis.has_major_gridlines = False
    chart_title(ch, "Every brick at its snap check")
    axis_title(ch.category_axis, "sideways error (mm)")
    axis_title(ch.value_axis, "height error (mm)")
    ch.category_axis.minimum_scale, ch.category_axis.maximum_scale, ch.category_axis.major_unit = 0, 2, 0.5
    ch.value_axis.minimum_scale, ch.value_axis.maximum_scale, ch.value_axis.major_unit = -1.5, 1.5, 0.5
    for v_ax in (ch.category_axis, ch.value_axis):
        v_ax.tick_labels.number_format, v_ax.tick_labels.number_format_is_linked = "0.0", False
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW  # x labels at the bottom, not on y = 0
    for ser, key in zip(ch.plots[0].series, ("accent", "orange")):
        ser.format.line.fill.background()
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 7
        ser.marker.format.fill.solid()
        setc(ser.marker.format.fill.fore_color, key)
        ser.marker.format.line.fill.background()
    bins = [i * 0.5 for i in range(11)]
    counts = [sum(lo <= e["tilt_deg"] < hi for e in rows) for lo, hi in zip(bins, bins[1:])]
    cd = CategoryChartData()
    cd.categories = ["%.1f" % hi for hi in bins[1:]]
    cd.add_series("bricks", counts)
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(rx + sw + 0.15), Inches(TOP),
                            Inches(rw - sw - 0.15), Inches(4.6), cd)
    gf.name = "Tilt histogram"
    ch = gf.chart
    style_chart(ch)
    chart_title(ch, "Tilt of every brick")
    axis_title(ch.category_axis, "tilt (°, bin upper edge)")
    axis_title(ch.value_axis, "bricks")
    ch.plots[0].gap_width = 15
    ch.plots[0].series[0].format.fill.solid()
    setc(ch.plots[0].series[0].format.fill.fore_color, "accent")
    caption(s, rx, TOP + 4.7, rw, "All 27 bricks: ≤ 1.46 mm sideways, ≤ 0.47 mm height, ≤ 1.85° "
            "tilt. Axes span the snap-check limits (2 mm, ±1.5 mm, 5°).", h=0.5, size=12)

    # 21 -- embedded video --------------------------------------------------------------
    s = content(prs, "Final run · Video", "The final run in 30 seconds")
    poster = Image.open(FIG / "fin_done_arch34.jpg")
    pw_, ph_ = poster.size
    ch_ = int(pw_ * 9 / 16)
    buf = io.BytesIO()
    poster.crop((0, (ph_ - ch_) // 2, pw_, (ph_ - ch_) // 2 + ch_)).save(buf, "JPEG", quality=90)
    buf.seek(0)
    vh = 4.6
    vw = vh * 16 / 9
    mv = s.shapes.add_movie(str(REPORT / "videos" / "dual_arch_s3_studfit_30s.mp4"), Inches((W - vw) / 2),
                            Inches(TOP), Inches(vw), Inches(vh), poster_frame_image=buf, mime_type="video/mp4")
    mv.name = "Final run video"
    caption(s, LM, TOP + vh + 0.12, CW, "Run dual_arch_s3_studfit_007 · every brick pressed onto resisting "
            "studs; B holds the arch for the crown, A holds S3 for steps 13–15", size=12)

    # 22 --------------------------------------------------------------------------------
    s = content(prs, "Results", "The two structures, from their inputs")
    for i, (img, cap) in enumerate((("sim_sf_fin_arch.jpg", "Arch, from a drawing — 11/11 bricks, 1 support"),
                                    ("sim_sf_fin_s3.jpg", "S3 bridge, from a brick list — 16/16 bricks, "
                                                          "3 supports"))):
        x = LM + i * (cw2 + 0.3)
        pic(s, img, x, TOP + 0.05, cw2, 4.5, cover=True)
        caption(s, x, TOP + 4.65, cw2, cap, size=14)

    # 23 --------------------------------------------------------------------------------
    s = content(prs, "Limitations", "What this does not show yet")
    card(s, LM, TOP + 0.1, cw2, 3.3,
         ["**Simulation only** — no physical robot.",
          "**Brick position comes from the simulator**; a camera-based version exists but was not used.",
          "**Supported steps are judged more gently**: brick goes in at 0.85× force, break check paused, small "
          "overlaps next to the helper allowed."], fill="white", size=17, bullet="dot", after=10, pad=0.25,
         name="Card")
    card(s, LM + cw2 + 0.3, TOP + 0.1, cw2, 3.3,
         ["**Cautious force model** (safety factor 3); for the arch crown it cannot show the support's benefit.",
          "**Drawings** cannot show hidden hollows.",
          "**Checks in small steps**, backed by the physics contact audit."], fill="white", size=17, bullet="dot",
         after=10, pad=0.25, name="Card")
    why(s, 5.4, "Next:", "keep the break check on by filtering weld lock-on spikes, supported presses at full force, "
        "camera-based brick position, the real cell.", h=0.9, size=16)

    # 24 --------------------------------------------------------------------------------
    s = title_slide(prs, "Summary", ["Drawing → blueprint → plan → control →",
                                     "two finished structures"],
                    "Carving from shadows + wall-like tiling · layer-by-layer plan with careful grips and press "
                    "points · force-balance linear program for supports · cuRobo + straight contact moves "
                    "+ IK · press from above onto resisting studs · five-layer collision safety with space "
                    "booking · simple coordinator · snap check + independent check.", size=36)
    tb(s, LM + 0.3, 5.8, 6, 0.6, [[("Questions?", {"bold": True})]], size=24)

    prs.core_properties.title = "Dual-Arm LEGO Assembly"
    prs.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
