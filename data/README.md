# Data Preparation

For simplicity, we have standardized the project drive structure originally provided by **Tonkin**. To run our application successfully, please ensure your `data` folder follows the structure below:

```
data/
└── projects/
    ├── project_name1/
    ├── project_name2/
    └── project_name3/
        └── 6_Issued/
            ├── project_pdf1
            └── project_pdf2
```

## Folder Structure

- **`projects/`** - the root directory containing all individual project folders.

- **`project_name*/`** - each folder represents one project.

- **`6_Issued/`** - subfolder containing issued deliverables (mainly PDF reports).

## What the Application Uses

Our application primarily processes the PDFs inside each projects's `6_Issued` folder, as these deliverables contain the key technical and historical information needed for knowledge extraction.

## Example

If your project is called _"240365 - Mitcham Enviro Monitoring"_ and it has a PDF report called _"240365R02A_Lynton Landfill GME November 2024_FINAL_appendices_Optimized.pdf"_, the expected path for this report would be:

```
data/projects/240365 - Mitcham Enviro Monitoring/6_Issued/240365R02A_Lynton Landfill GME November 2024_FINAL_appendices_Optimized.pdf
```
