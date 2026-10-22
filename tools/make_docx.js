// Turn the glossary data from build_glossary.py into a Word document.
// Driven by: python3 tools/build_glossary.py
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, ShadingType, HeadingLevel, AlignmentType, BorderStyle,
  Footer, PageNumber, LevelFormat, convertInchesToTwip,
} = require("docx");

const [, , dataPath, outPath] = process.argv;
const D = JSON.parse(fs.readFileSync(dataPath, "utf8"));

const PAGE_W = 12240, PAGE_H = 15840, MARGIN = 1440;
const CONTENT = PAGE_W - 2 * MARGIN;          // 9360
const MONO = "Courier New";
const INK_SOFT = "6B7280";
const RULE = "D1D5DB";
const HEAD_FILL = "EEF2F7";

function p(opts) { return new Paragraph(opts); }

function body(text, extra) {
  return p(Object.assign({ children: [new TextRun({ text, size: 20 })], spacing: { after: 100 } }, extra || {}));
}

function soft(text, extra) {
  return p(Object.assign({
    children: [new TextRun({ text, size: 18, color: INK_SOFT })],
    spacing: { after: 80 },
  }, extra || {}));
}

// Code can contain newlines; Word needs one paragraph per line.
function codeLines(text, opts) {
  const o = opts || {};
  return String(text).split("\n").map(function (line, i) {
    return p({
      children: [new TextRun({ text: line, font: MONO, size: 18, color: o.color || "1F2937" })],
      spacing: { after: i === String(text).split("\n").length - 1 ? (o.after === undefined ? 80 : o.after) : 0 },
    });
  });
}

function labelled(label, value, italic) {
  return p({
    children: [
      new TextRun({ text: label + " ", bold: true, size: 18, color: INK_SOFT }),
      new TextRun({ text: value, size: 18, italics: !!italic }),
    ],
    spacing: { after: 80 },
  });
}

function cell(children, width, opts) {
  const o = opts || {};
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: o.fill ? { type: ShadingType.CLEAR, color: "auto", fill: o.fill } : undefined,
    margins: { top: 90, bottom: 90, left: 120, right: 120 },
    children: children,
  });
}

function headerCell(text, width) {
  return cell([p({
    children: [new TextRun({ text: text, bold: true, size: 18 })],
  })], width, { fill: HEAD_FILL });
}

function table(columnWidths, rows) {
  return new Table({
    columnWidths: columnWidths,
    width: { size: CONTENT, type: WidthType.DXA },
    rows: rows,
  });
}

function hr() {
  return p({
    text: "",
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 1 } },
    spacing: { after: 220 },
  });
}

function h1(text) {
  return p({ text: text, heading: HeadingLevel.HEADING_1, spacing: { before: 380, after: 160 } });
}

function h2(text) {
  return p({ text: text, heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 } });
}

// ------------------------------------------------------------------ content
const kids = [];

kids.push(p({ text: D.title, heading: HeadingLevel.TITLE, spacing: { after: 80 } }));
kids.push(body(D.subtitle));
kids.push(soft("Covers week " + D.weeks_label + ", which is " + D.n_problems +
               " problems and " + D.n_functions + " functions. Generated " +
               D.generated + " from " + D.repo + "."));
kids.push(hr());

kids.push(h1("How to use this"));
kids.push(body("This is a lookup, not a tutorial. Each entry says what the function does, " +
               "shows one line of real use, and names the mistake it tends to invite. " +
               "The tutorials are the foundations notebook and the How it works section of " +
               "each problem."));
kids.push(body("The right hand tag on each entry says which problem introduces it, so you " +
               "can see what you have already met. Entries tagged Foundations appear in the " +
               "foundations notebook rather than in a dated problem."));
kids.push(body("This document is generated. It is rebuilt every time a week is added, so it " +
               "always matches the notebooks. Do not edit it by hand, because the next " +
               "rebuild would overwrite your changes."));

kids.push(h2("Contents"));
["1. What is covered so far",
 "2. Python essentials",
 "3. numpy",
 "4. pandas",
 "5. scipy",
 "6. matplotlib",
 "7. scikit-image and OpenCV",
 "8. scikit-learn",
 "9. The house rules",
 "10. Error messages, and what they mean"].forEach(function (line) {
  kids.push(soft(line, { spacing: { after: 40 } }));
});

// --- 1. coverage
kids.push(h1("1. What is covered so far"));
const covRows = [new TableRow({
  tableHeader: true,
  children: [headerCell("Problem", 1200), headerCell("Topic", 6360), headerCell("New", 1800)],
})];
D.problems.forEach(function (pr) {
  covRows.push(new TableRow({
    children: [
      cell([p({ children: [new TextRun({ text: pr.pid, font: MONO, bold: true, size: 18 })] })], 1200),
      cell([body(pr.title, { spacing: { after: 40 } }), soft(pr.topic, { spacing: { after: 0 } })], 6360),
      cell([p({ children: [new TextRun({ text: pr.n_new + " new", size: 18, color: INK_SOFT })] })], 1800),
    ],
  }));
});
kids.push(table([1200, 6360, 1800], covRows));

// --- 2 to 5. the function tables
const SECTION_TITLES = {
  "Python": "2. Python essentials",
  "numpy": "3. numpy",
  "pandas": "4. pandas",
  "scipy": "5. scipy",
  "matplotlib": "6. matplotlib",
  "imaging": "7. scikit-image and OpenCV",
  "sklearn": "8. scikit-learn",
};

D.groups.forEach(function (group) {
  kids.push(h1(SECTION_TITLES[group.name] || group.name));
  const rows = [new TableRow({
    tableHeader: true,
    children: [headerCell("Function", 2400), headerCell("What it does", 6960)],
  })];
  group.rows.forEach(function (r) {
    const left = [
      p({
        children: [new TextRun({ text: r.key, font: MONO, bold: true, size: 18 })],
        spacing: { after: 60 },
      }),
      p({
        children: [new TextRun({ text: r.seen, size: 15, color: INK_SOFT })],
        spacing: { after: 0 },
      }),
    ];
    const right = [];
    right.push(p({
      children: [new TextRun({ text: r.sig, font: MONO, size: 18, color: "1F4E79" })],
      spacing: { after: 90 },
    }));
    right.push(body(r.what, { spacing: { after: r.example ? 90 : 0 } }));
    if (r.example) {
      right.push(p({
        children: [new TextRun({ text: "Example", bold: true, size: 15, color: INK_SOFT })],
        spacing: { after: 30 },
      }));
      codeLines(r.example, { after: r.watch ? 90 : 0 }).forEach(function (c) { right.push(c); });
    }
    if (r.watch) {
      right.push(p({
        children: [
          new TextRun({ text: "Watch out  ", bold: true, size: 15, color: "9A3412" }),
          new TextRun({ text: r.watch, size: 18, italics: true }),
        ],
        spacing: { after: 0 },
      }));
    }
    rows.push(new TableRow({ children: [cell(left, 2400), cell(right, 6960)] }));
  });
  kids.push(table([2400, 6960], rows));
});

// --- 6. house rules
kids.push(h1("9. The house rules"));
kids.push(body("These are enforced by the check() cell in every problem, not just mentioned. " +
               "If you break one, the check names it."));
D.rules.forEach(function (r) {
  kids.push(p({
    children: [new TextRun({ text: r.rule, bold: true, size: 20 })],
    spacing: { before: 180, after: 60 },
  }));
  kids.push(body(r.why));
});

// --- 7. errors
kids.push(h1("10. Error messages, and what they mean"));
kids.push(body("Read the LAST line of a traceback first. It names the problem. The lines " +
               "above it only say where it happened."));
const errRows = [new TableRow({
  tableHeader: true,
  children: [headerCell("What you see", 4200), headerCell("What it usually means", 5160)],
})];
D.errors.forEach(function (e) {
  errRows.push(new TableRow({
    children: [
      cell(codeLines(e.msg, { after: 0, color: "9A3412" }), 4200),
      cell([body(e.means, { spacing: { after: 0 } })], 5160),
    ],
  }));
});
kids.push(table([4200, 5160], errRows));

kids.push(p({ text: "", spacing: { before: 300 } }));
kids.push(soft("Generated by tools/build_glossary.py from tools/function_catalog.py. " +
               "Rebuilt whenever a week is added."));

// ------------------------------------------------------------------ document
const doc = new Document({
  creator: "internship-prep",
  title: D.title,
  description: D.subtitle,
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 20 } },
      heading1: { run: { font: "Calibri", size: 30, bold: true, color: "1F4E79" } },
      heading2: { run: { font: "Calibri", size: 24, bold: true, color: "1F4E79" } },
      title: { run: { font: "Calibri", size: 48, bold: true, color: "16181D" } },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
      },
    },
    footers: {
      default: new Footer({
        children: [p({
          alignment: AlignmentType.RIGHT,
          children: [
            new TextRun({ text: "Function Reference, week " + D.weeks_label + "    ", size: 16, color: INK_SOFT }),
            new TextRun({ children: [PageNumber.CURRENT], size: 16, color: INK_SOFT }),
          ],
        })],
      }),
    },
    children: kids,
  }],
});

Packer.toBuffer(doc).then(function (buf) {
  fs.writeFileSync(outPath, buf);
  console.log("wrote " + outPath + " (" + Math.round(buf.length / 1024) + " KB)");
}).catch(function (err) {
  console.error(err);
  process.exit(1);
});
