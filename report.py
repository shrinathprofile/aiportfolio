import streamlit as st
import ollama
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime
import os

# Streamlit page configuration
st.set_page_config(
    page_title="Automated Report Generator",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

class ReportGenerator:
    def __init__(self):
        self.model = "llama3.2:1b"  # Adjust based on your Ollama setup
        self.output_dir = "temp"
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def analyze_data(self, df):
        """Analyze the dataset and return key statistics"""
        analysis = {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_cols": df.select_dtypes(exclude=[np.number]).columns.tolist(),
            "summary_stats": df.describe().to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        return analysis
    
    def generate_visualizations(self, df):
        """Generate a set of generalized, insightful visualizations for any dataset."""
        viz_files = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # 1. Correlation Heatmap (Numeric Columns)
    # Insight: Shows relationships between numeric variables (e.g., satisfaction vs. performance)
        if len(numeric_cols) >= 2:
            try:
                plt.figure(figsize=(10, 6))
                correlation_matrix = df[numeric_cols].corr()
                plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
                plt.colorbar(label='Correlation Coefficient')
                plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha='right')
                plt.yticks(range(len(numeric_cols)), numeric_cols)
                plt.title('Correlation Heatmap of Numeric Variables')
                file_path = os.path.join(self.output_dir, 'correlation_heatmap.png')
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()
                viz_files.append(file_path)
            except Exception as e:
                st.warning(f"Could not generate correlation heatmap: {str(e)}")

    # 2. Distribution of a Key Numeric Column (e.g., Satisfaction, Income)
    # Insight: Highlights spread and outliers in a critical metric
        key_numeric = None
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in ['satisfaction', 'score', 'rate', 'income']):
                key_numeric = col
                break
        if not key_numeric and numeric_cols:  # Fallback to highest-variance column
            key_numeric = max(numeric_cols, key=lambda x: df[x].var(), default=None)
        if key_numeric:
            try:
                plt.figure(figsize=(10, 6))
                plt.hist(df[key_numeric].dropna(), bins=20, alpha=0.7, color='blue')
                plt.axvline(df[key_numeric].mean(), color='r', linestyle='--', label=f'Mean: {df[key_numeric].mean():.2f}')
                plt.title(f'Distribution of {key_numeric}')
                plt.xlabel(key_numeric)
                plt.ylabel('Frequency')
                plt.legend()
                file_path = os.path.join(self.output_dir, f'{key_numeric}_histogram.png')
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()
                viz_files.append(file_path)
            except Exception as e:
                st.warning(f"Could not generate histogram for {key_numeric}: {str(e)}")

    # 3. Box Plot of Numeric vs. Categorical (e.g., Salary by Department)
    # Insight: Compares a key metric across groups
        key_categorical = None
        for col in categorical_cols:
            if df[col].nunique() <= 10:  # Limit to low-cardinality columns
                key_categorical = col
                break
        if key_categorical and key_numeric:
            try:
                plt.figure(figsize=(10, 6))
                df.boxplot(column=key_numeric, by=key_categorical, grid=False)
                plt.title(f'{key_numeric} by {key_categorical}')
                plt.xlabel(key_categorical)
                plt.ylabel(key_numeric)
                plt.xticks(rotation=45, ha='right')
                file_path = os.path.join(self.output_dir, f'{key_numeric}_by_{key_categorical}_box.png')
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()
                viz_files.append(file_path)
            except Exception as e:
                st.warning(f"Could not generate box plot for {key_numeric} by {key_categorical}: {str(e)}")

    # 4. Trend of a Time-Related Numeric Column
    # Insight: Reveals patterns over time or index
        time_col = None
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'month', 'year']) or df.index.is_numeric():
                time_col = col
                break
        if time_col and time_col in df.columns:
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(df[time_col], df[key_numeric], label=key_numeric)
                plt.title(f'Trend of {key_numeric} over {time_col}')
                plt.xlabel(time_col)
                plt.ylabel(key_numeric)
                plt.legend()
                file_path = os.path.join(self.output_dir, f'{key_numeric}_trend.png')
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()
                viz_files.append(file_path)
            except Exception as e:
                st.warning(f"Could not generate trend chart: {str(e)}")
        elif key_numeric:  # Fallback to index-based trend
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df[key_numeric], label=key_numeric)
                plt.title(f'Trend of {key_numeric} over Index')
                plt.xlabel('Index')
                plt.ylabel(key_numeric)
                plt.legend()
                file_path = os.path.join(self.output_dir, f'{key_numeric}_index_trend.png')
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()
                viz_files.append(file_path)
            except Exception as e:
                st.warning(f"Could not generate index trend chart: {str(e)}")

    # 5. Stacked Bar of Categorical vs. Binary Outcome
    # Insight: Shows proportions (e.g., attrition by category)
        binary_col = next((col for col in categorical_cols if df[col].nunique() == 2), None)
        if binary_col and key_categorical and binary_col != key_categorical:
            try:
                plt.figure(figsize=(10, 6))
                crosstab = pd.crosstab(df[key_categorical], df[binary_col])
                crosstab.plot(kind='bar', stacked=True)
                plt.title(f'{binary_col} by {key_categorical}')
                plt.xlabel(key_categorical)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                file_path = os.path.join(self.output_dir, f'{binary_col}_by_{key_categorical}_stacked.png')
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()
                viz_files.append(file_path)
            except Exception as e:
                st.warning(f"Could not generate stacked bar chart: {str(e)}")

        return viz_files
    
    def generate_narrative(self, analysis):
        """Generate a strategic, executive-level narrative tailored for C-level audiences."""
        prompt = f"""
        As a senior business analyst reporting to C-level executives, create a compelling and concise report narrative based on the provided dataset analysis. Use a professional, authoritative, and forward-looking tone suitable for CEOs, CFOs, and COOs. Follow this strict structure and guidelines:

        1. **Executive Summary** (150-200 words)
            - Provide a high-level overview of organizational performance or key dataset trends.
            - Highlight 1-2 major strengths (e.g., high employee satisfaction, revenue growth) and 1-2 opportunities (e.g., attrition, process inefficiencies) using specific metrics from the analysis (e.g., average MonthlyIncome: {analysis['summary_stats'].get('MonthlyIncome', {}).get('mean', 'N/A')}).
            - Avoid technical jargon; focus on business impact (e.g., "This positions us for market leadership").
            - Set the stage for strategic decision-making.

        2. **Key Findings** (250-300 words)
            - Present 3-5 strategic insights derived from the data.
            - Each finding should include:
                - A clear, business-focused statement (e.g., "Employee satisfaction drives retention").
                - Evidence from summary statistics (e.g., JobSatisfaction mean: {analysis['summary_stats'].get('JobSatisfaction', {}).get('mean', 'N/A')}) or trends.
                - A high-level implication (e.g., "This could enhance customer loyalty by X%").
            - Prioritize metrics relevant to C-level priorities (e.g., income, satisfaction, attrition) over technical details like missing values unless they significantly impact analysis.
            - Use bullet points for clarity.

        3. **Strategic Recommendations** (200-250 words)
            - Offer 3-5 actionable, measurable recommendations tied to Key Findings.
            - Each recommendation should:
                - Address a specific finding (e.g., "To boost satisfaction, we recommend...").
                - Include a clear goal (e.g., "Increase satisfaction to 4/5 within 6 months").
                - Suggest resource allocation or initiatives (e.g., "Invest $X in training programs").
            - Use a decisive tone (e.g., "We will prioritize...", "Implementation should begin in Q2").

        4. **Conclusion** (100-150 words)
            - Summarize the organization’s strategic position based on strengths and opportunities.
            - Reinforce confidence in the recommendations with a forward-looking statement (e.g., "This strategy will drive 10% growth by year-end").
            - End with a call to action (e.g., "Approve the plan by next board meeting").

        **Dataset Summary:**
        - Rows: {analysis['rows']}
        - Columns: {analysis['columns']}
        - Numeric Columns: {', '.join(analysis['numeric_cols'])}
        - Categorical Columns: {', '.join(analysis['categorical_cols'])}
        - Summary Statistics: {analysis['summary_stats']}
        - Missing Values: {analysis['missing_values']} (only note if >10% of data is missing per column)

        **Guidelines:**
        - Focus on business outcomes (e.g., profitability, engagement) rather than technical details.
        - Limit mention of missing values to cases where they obscure major trends (e.g., >10% missing).
        - Use data to support claims but avoid overloading with numbers; aim for 2-3 key metrics per section.
        - Keep paragraphs to 3-5 sentences; use bullet points for lists.
        - Ensure the narrative is standalone and compelling without requiring external context.

        Output the narrative in plain text, with section headers marked as **Section Title** (e.g., **Executive Summary**).
        """

        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response']
    
    def create_pdf_report(self, narrative, viz_files, df, filename="report.pdf"):
        """Generate a professional PDF report with enhanced layout and styling."""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.75*inch, leftMargin=0.75*inch, 
                            topMargin=1*inch, bottomMargin=0.75*inch)
        styles = getSampleStyleSheet()

    # Custom styles for better typography
        styles.add(ParagraphStyle(name='TitleCustom', fontSize=18, leading=22, textColor=colors.HexColor('#003087'), spaceAfter=12))
        styles.add(ParagraphStyle(name='Subtitle', fontSize=14, leading=18, textColor=colors.HexColor('#005566'), spaceAfter=10))
        styles.add(ParagraphStyle(name='BodyCustom', fontSize=10, leading=14, spaceAfter=8))
        styles.add(ParagraphStyle(name='Caption', fontSize=8, leading=10, textColor=colors.grey, alignment=1))  # Centered captions

        story = []

    # Header function for all pages
        def add_header(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 8)
            canvas.setFillColor(colors.grey)
            canvas.drawString(0.75*inch, doc.pagesize[1] - 0.5*inch, f"Executive Data Report - Generated on {datetime.now().strftime('%Y-%m-%d')}")
            canvas.line(0.75*inch, doc.pagesize[1] - 0.6*inch, doc.pagesize[0] - 0.75*inch, doc.pagesize[1] - 0.6*inch)
            canvas.restoreState()

    # Footer function
        def add_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 8)
            canvas.setFillColor(colors.grey)
            canvas.drawString(0.75*inch, 0.5*inch, f"Page {doc.page}")
            canvas.restoreState()

    # Title Page
        story.append(Paragraph("Executive Data Report", styles['TitleCustom']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", styles['BodyCustom']))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Prepared by: Automated Report Generator", styles['BodyCustom']))
        story.append(PageBreak())

    # Narrative Sections
        sections = narrative.split('\n\n')
        current_section = None
        for para in sections:
            if para.startswith('"') and para.endswith('"') and len(para.strip('"')) < 50:  # Detect section headers
                if current_section:  # Add spacing before new section
                    story.append(Spacer(1, 0.25*inch))
                current_section = para.strip('"')
                story.append(Paragraph(current_section, styles['Subtitle']))
            else:
                story.append(Paragraph(para, styles['BodyCustom']))

    # Visualizations Section
        if viz_files:
            story.append(PageBreak())
            story.append(Paragraph("Data Visualizations", styles['Subtitle']))
            story.append(Spacer(1, 0.2*inch))
            for i, viz_file in enumerate(viz_files, 1):
                if os.path.exists(viz_file):
                    img = Image(viz_file, width=6.5*inch, height=4*inch)  # Fit within margins
                    story.append(img)
                    story.append(Paragraph(f"Figure {i}: {os.path.basename(viz_file).replace('_', ' ').replace('.png', '').title()}", 
                                     styles['Caption']))
                    story.append(Spacer(1, 0.15*inch))

    # Data Summary Section
        story.append(PageBreak())
        story.append(Paragraph("Data Summary", styles['Subtitle']))
        story.append(Spacer(1, 0.2*inch))
        summary_data = df.describe().reset_index()
        table_data = [summary_data.columns.tolist()] + summary_data.values.tolist()
        table = Table(table_data, colWidths=[1.2*inch] + [0.9*inch]*(len(summary_data.columns)-1))
        table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003087')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F6F5')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ])
        story.append(table)

    # Build the document with header/footer
        doc.build(story, onFirstPage=add_header, onLaterPages=lambda c, d: (add_header(c, d), add_footer(c, d)))
        buffer.seek(0)
        return buffer

    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))

def main():
    st.title("📈 Automated Report Generator")
    st.markdown("Generate professional reports for C-level executives from your data")
    
    # Initialize report generator
    generator = ReportGenerator()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Report Settings")
        report_title = st.text_input("Report Title", "Executive Data Insights")
        include_viz = st.checkbox("Include Visualizations", True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your data (CSV, Excel)", 
                                  type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Show preview
            st.subheader("Data Preview")
            st.write(df.head())
            
            if st.button("Generate Report"):
                with st.spinner("Generating your executive report..."):
                    # Analyze data
                    analysis = generator.analyze_data(df)
                    
                    # Generate narrative
                    narrative = generator.generate_narrative(analysis)
                    
                    # Generate visualizations
                    viz_files = []
                    if include_viz:
                        viz_files = generator.generate_visualizations(df)
                    
                    # Create PDF
                    pdf_buffer = generator.create_pdf_report(narrative, viz_files, df)
                    
                    # Display narrative
                    st.subheader("Report Preview")
                    st.markdown(narrative)
                    
                    # Download button
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"{report_title}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            generator.cleanup()  # Clean up temporary files
    
    # Instructions
    st.markdown("""
    ### How to Use:
    1. Upload your CSV or Excel file
    2. Customize report settings in the sidebar
    3. Click 'Generate Report'
    4. Review the preview and download the PDF
    """)

if __name__ == "__main__":
    main()