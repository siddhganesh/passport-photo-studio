# Passport Photo Print Studio

A single self-contained web tool: upload any phone photo (gallery or WhatsApp), it auto-enhances the image (color/contrast correction, sharpening, upscale to 300 DPI), crops it to a standard passport or stamp size, and lays out multiple copies on a 4×6" or A4 sheet with cut guides — ready to print.

## How to use it
Just open **`index.html`** in any web browser (double-click it). No install, no internet connection, and no server required — everything runs locally in the browser, and nothing is uploaded anywhere.

## Deploying it online (optional)
Because the tool is named `index.html`, this whole folder can be dragged straight into Netlify or Vercel (or pushed to GitHub Pages) to get it a public link, if you'd rather share a URL than the file itself.

## Customizing
Everything — colors, paper sizes, margins, DPI — lives in the `<style>` and `<script>` sections at the top and bottom of `index.html`. Key constants to tweak are near the top of the script:

- `MARGIN_MM` / `GAP_MM` — outer margin and spacing between photos on the sheet
- `PAPER_PRESETS` / `PHOTO_PRESETS` — add more paper or photo sizes here
- `DPI` — output resolution (300 is standard print quality)
