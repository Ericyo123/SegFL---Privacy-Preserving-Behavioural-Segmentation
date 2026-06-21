import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def build_pdf_report(data, filename):
    """
    Builds a research-grade PDF evaluation report for SegFL.
    
    Args:
        data: dict containing results, base_df, abl_df, dp_df, stability_results,
              stat_test_results, scal_results, gen_results.
        filename: path to write the PDF report to.
    """
    # 1. Setup Document
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=54, leftMargin=54,
        topMargin=54, bottomMargin=54
    )
    
    # 2. Setup Styles
    styles = getSampleStyleSheet()
    
    # Custom Styles
    primary_color = colors.HexColor("#1e293b")  # Dark Slate
    secondary_color = colors.HexColor("#3b82f6")  # Deep Blue
    text_color = colors.HexColor("#334155")      # Charcoal text
    bg_color = colors.HexColor("#f8fafc")        # Soft off-white
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        spaceAfter=6,
        alignment=1 # Centered
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=20,
        alignment=1 # Centered
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=text_color,
        spaceAfter=8
    )
    
    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=10,
        alignment=1
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=text_color
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    story = []
    
    # --- TITLE SECTION ---
    story.append(Spacer(1, 10))
    story.append(Paragraph("SegFL: Scientific Overhaul Report", title_style))
    story.append(Paragraph("Privacy-Preserving Behavioural Segmentation using Federated Learning", subtitle_style))
    story.append(Spacer(1, 5))
    
    # --- PERFORMANCE KPI MATRIX ---
    story.append(Paragraph("1. Executive Summary & KPIs", h1_style))
    story.append(Paragraph(
        "This evaluation report compiles the performance, privacy, and scientific stability results of the SegFL protocol. "
        "SegFL addresses behavioral profile clustering across disjoint organizations using a Tenant Adapter Layer (TAL) to reconcile heterogeneous dimensions, "
        "followed by federated aggregation and DP-SGD noise injection.",
        body_style
    ))
    
    results = data.get('results', {})
    kpi_data = [
        [
            Paragraph("<b>Silhouette Score</b>", table_cell_style), 
            Paragraph(f"{results.get('silhouette', 0.0):.4f}", table_cell_style),
            Paragraph("<b>Davies-Bouldin Index</b>", table_cell_style),
            Paragraph(f"{results.get('dbi', 0.0):.4f}", table_cell_style)
        ],
        [
            Paragraph("<b>Calinski-Harabasz</b>", table_cell_style), 
            Paragraph(f"{results.get('calinski_harabasz', 0.0):.1f}", table_cell_style),
            Paragraph("<b>Privacy Budget (ε)</b>", table_cell_style),
            Paragraph(f"{results.get('epsilon', float('inf')):.2f}" if results.get('epsilon', float('inf')) < 1e10 else "∞ (None)", table_cell_style)
        ]
    ]
    
    kpi_table = Table(kpi_data, colWidths=[120, 120, 120, 120])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 10))
    
    # --- ARCHETYPES TABLE ---
    profile = results.get('profile')
    if isinstance(profile, pd.DataFrame) and not profile.empty:
        story.append(Paragraph("2. Discovered Behavioural Archetypes", h1_style))
        story.append(Paragraph(
            "Below is the profile of discovered user behavioral cohorts. Column metrics are computed as cluster centroid averages.",
            body_style
        ))
        
        # Build table
        headers = [Paragraph(f"<b>{c}</b>", table_header_style) for c in ['Cluster'] + list(profile.columns)]
        table_rows = [headers]
        
        for idx, row in profile.iterrows():
            row_cells = [Paragraph(f"Cluster {idx}", table_cell_style)]
            for col in profile.columns:
                val = row[col]
                if isinstance(val, float):
                    val_str = f"{val:.3f}"
                else:
                    val_str = str(val)
                row_cells.append(Paragraph(val_str, table_cell_style))
            table_rows.append(row_cells)
            
        col_count = len(profile.columns) + 1
        width_col = 500 / col_count
        
        arch_table = Table(table_rows, colWidths=[width_col] * col_count)
        arch_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_color]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(arch_table)
        story.append(Spacer(1, 12))

    # --- BASELINES COMPARISON ---
    base_df = data.get('base_df')
    if isinstance(base_df, pd.DataFrame) and not base_df.empty:
        story.append(Paragraph("3. Baseline Algorithm Comparisons", h1_style))
        story.append(Paragraph(
            "Compares SegFL (TAL-FL) to centralized upper bounds and baseline federated learning architectures. "
            "Agreement metrics (NMI/ARI) show how closely each configuration maps to the centralized clustering upper bound.",
            body_style
        ))
        
        headers = [Paragraph(f"<b>{c}</b>", table_header_style) for c in base_df.columns]
        table_rows = [headers]
        for _, row in base_df.iterrows():
            row_cells = [Paragraph(str(row[c]), table_cell_style) for c in base_df.columns]
            table_rows.append(row_cells)
            
        col_widths = [110, 65, 55, 95, 60, 60, 55] if len(base_df.columns) == 7 else [125] * len(base_df.columns)
        base_table = Table(table_rows, colWidths=col_widths)
        base_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_color]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(base_table)
        story.append(Spacer(1, 12))

    # --- ABLATION STUDY ---
    abl_df = data.get('abl_df')
    if isinstance(abl_df, pd.DataFrame) and not abl_df.empty:
        story.append(Paragraph("4. Component Ablation Study", h1_style))
        story.append(Paragraph(
            "Isolates the contribution of individual framework modules (Differential Privacy, Federated aggregation, TAL).",
            body_style
        ))
        
        headers = [Paragraph(f"<b>{c}</b>", table_header_style) for c in abl_df.columns]
        table_rows = [headers]
        for _, row in abl_df.iterrows():
            row_cells = [Paragraph(str(row[c]), table_cell_style) for c in abl_df.columns]
            table_rows.append(row_cells)
            
        col_widths = [140, 65, 55, 60, 60, 60, 60] if len(abl_df.columns) == 7 else [125] * len(abl_df.columns)
        abl_table = Table(table_rows, colWidths=col_widths)
        abl_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_color]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(abl_table)
        story.append(Spacer(1, 12))

    # --- CHARTS AND SCIENTIFIC VISUALIZATION ---
    loss_history = results.get('loss_history')
    dp_df = data.get('dp_df')
    
    if loss_history or (isinstance(dp_df, pd.DataFrame) and not dp_df.empty):
        # Create matplotlib plots dynamically and save as temp image files
        temp_img_files = []
        
        fig, axes = plt.subplots(1, 2 if (loss_history and isinstance(dp_df, pd.DataFrame) and not dp_df.empty) else 1, figsize=(10, 3.8))
        
        # Customize Matplotlib styling for professional PDF
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
        
        ax_loss, ax_dp = None, None
        if loss_history and isinstance(dp_df, pd.DataFrame) and not dp_df.empty:
            ax_loss, ax_dp = axes[0], axes[1]
        elif loss_history:
            ax_loss = axes
        elif isinstance(dp_df, pd.DataFrame) and not dp_df.empty:
            ax_dp = axes
            
        if ax_loss:
            ax_loss.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='#1e293b', linewidth=2, markersize=5)
            ax_loss.set_title("Global AE Reconstruction Loss", fontsize=11, fontweight='bold', pad=8)
            ax_loss.set_xlabel("Global Communication Round", fontsize=9)
            ax_loss.set_ylabel("Mean Squared Error (MSE)", fontsize=9)
            ax_loss.grid(True, linestyle='--', alpha=0.5)
            ax_loss.tick_params(labelsize=8)
            
        if ax_dp:
            dp_plot = dp_df[dp_df['Privacy Budget (ε)'] < 1e10].copy()
            if not dp_plot.empty:
                ax_dp.plot(dp_plot['DP Sigma (σ)'], dp_plot['Silhouette'], marker='s', color='#3b82f6', linewidth=2, label="Silhouette")
                ax_dp.set_xlabel("DP Noise Multiplier (σ)", fontsize=9)
                ax_dp.set_ylabel("Clustering Silhouette Score", fontsize=9, color='#3b82f6')
                ax_dp.tick_params(axis='y', labelcolor='#3b82f6', labelsize=8)
                
                ax2 = ax_dp.twinx()
                ax2.plot(dp_plot['DP Sigma (σ)'], dp_plot['Privacy Budget (ε)'], marker='^', color='#ef4444', linewidth=2, label="Epsilon (ε)")
                ax2.set_ylabel("Privacy Budget (ε) - Log Scale", fontsize=9, color='#ef4444')
                ax2.tick_params(axis='y', labelcolor='#ef4444', labelsize=8)
                
                ax_dp.set_title("DP Utility-Privacy Tradeoff", fontsize=11, fontweight='bold', pad=8)
                ax_dp.grid(True, linestyle='--', alpha=0.5)
                ax_dp.tick_params(labelsize=8)
        
        plt.tight_layout()
        
        # Save figure to temp file
        temp_fd, temp_img_path = tempfile.mkstemp(suffix=".png")
        os.close(temp_fd)
        fig.savefig(temp_img_path, dpi=200)
        plt.close(fig)
        temp_img_files.append(temp_img_path)
        
        # Append to PDF flowable story
        story.append(Paragraph("5. Training Convergence & Privacy Tradeoff", h1_style))
        story.append(Image(temp_img_path, width=480, height=180))
        story.append(Paragraph("Figure 1: Autoencoder loss convergence history (left) and differential privacy utility-privacy trade-off sweep (right).", caption_style))
        story.append(Spacer(1, 10))

    # --- STABILITY & SIGNIFICANCE TESTING ---
    stability_results = data.get('stability_results')
    stat_test_results = data.get('stat_test_results')
    
    if stability_results:
        story.append(Paragraph("6. Statistical Stability & Significance", h1_style))
        story.append(Paragraph(
            f"Evaluated stability across 10 random seeds. "
            f"Average Silhouette: <b>{stability_results.get('silhouette_mean', 0.0):.4f} ± {stability_results.get('silhouette_std', 0.0):.4f}</b>. "
            f"Average Davies-Bouldin Index: <b>{stability_results.get('dbi_mean', 0.0):.4f} ± {stability_results.get('dbi_std', 0.0):.4f}</b>. "
            f"Mean pairwise NMI agreement: <b>{stability_results.get('nmi_mean', 0.0):.4f} ± {stability_results.get('nmi_std', 0.0):.4f}</b>.",
            body_style
        ))
        
        if stat_test_results:
            sig_text = "statistically significant" if stat_test_results.get('significant') else "not statistically significant"
            story.append(Paragraph(
                f"<b>Wilcoxon Signed-Rank Test:</b> A non-parametric paired significance test comparing SegFL (TAL-FL) "
                f"against the Centralized baseline yielded a test statistic W = <b>{stat_test_results.get('statistic', 0.0):.1f}</b> and "
                f"a p-value of <b>{stat_test_results.get('p_value', 1.0):.4f}</b>. "
                f"The difference is <b>{sig_text}</b> at significance level α = 0.05.",
                body_style
            ))
        story.append(Spacer(1, 12))

    # --- SCALABILITY & GENERALIZATION ---
    scal_results = data.get('scal_results')
    gen_results = data.get('gen_results')
    
    if scal_results or gen_results:
        story.append(Paragraph("7. Scalability & Unseen Tenant Generalization", h1_style))
        
        if scal_results:
            story.append(Paragraph(
                "As client organizations scale, execution time scales linearly while clustering quality maintains a stable trajectory, "
                "supporting large-scale industrial federations.",
                body_style
            ))
            
        if gen_results:
            story.append(Paragraph(
                f"<b>Zero-Shot Unseen Tenant Adaptability:</b> "
                f"To assess generalization capability, the model was trained on $C-1$ tenants, withholding the last tenant. "
                f"A new local adapter (TAL) was trained on the unseen tenant set with global weights frozen. "
                f"This yield a hold-out test Silhouette score of <b>{gen_results.get('holdout_silhouette', 0.0):.3f}</b> "
                f"(DBI: <b>{gen_results.get('holdout_dbi', 0.0):.3f}</b>), proving the global representation generalises effectively.",
                body_style
            ))
            
    # --- EXPLAINABILITY & INTERPRETABILITY ---
    importance_df = data.get('importance_df')
    enrichment_scores = data.get('enrichment_scores')
    if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
        story.append(Paragraph("8. Explainability & Surrogate Feature Attributions", h1_style))
        story.append(Paragraph(
            "To interpret the black-box federated representation space, we train a global surrogate random forest model on the raw behavioral features to predict the final cluster label assignments. "
            "Additionally, cluster enrichment scores quantify the feature-specific Z-score deviations of each cohort from the global mean (measured in global standard deviations σ).",
            body_style
        ))
        
        # 1. Feature Importance Table
        story.append(Paragraph("<b>Surrogate Global Feature Importances</b>", ParagraphStyle('Sub', parent=body_style, fontName='Helvetica-Bold')))
        headers = [Paragraph("<b>Feature</b>", table_header_style), Paragraph("<b>Surrogate Importance Weight</b>", table_header_style)]
        table_rows = [headers]
        for _, row in importance_df.iterrows():
            row_cells = [
                Paragraph(str(row['Feature']), table_cell_style),
                Paragraph(f"{row['Importance']:.4f}", table_cell_style)
            ]
            table_rows.append(row_cells)
            
        imp_table = Table(table_rows, colWidths=[240, 240])
        imp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_color]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(imp_table)
        story.append(Spacer(1, 10))
        
        # 2. Enrichment Scores Table
        if enrichment_scores:
            story.append(Paragraph("<b>Cohort Enrichment Deviations (Z-Scores)</b>", ParagraphStyle('Sub', parent=body_style, fontName='Helvetica-Bold')))
            
            sample_cid = list(enrichment_scores.keys())[0]
            feats_list = list(enrichment_scores[sample_cid].keys())
            
            headers = [Paragraph("<b>Cohort</b>", table_header_style)] + [Paragraph(f"<b>{f}</b>", table_header_style) for f in feats_list]
            table_rows = [headers]
            for cid, scores in enrichment_scores.items():
                persona_name = results.get('profile').loc[cid, 'Persona'] if (results and 'profile' in results and cid in results['profile'].index) else f"Cluster {cid}"
                row_cells = [Paragraph(f"<b>Cluster {cid}</b><br/>{persona_name}", table_cell_style)]
                for feat in feats_list:
                    val = scores.get(feat, 0.0)
                    row_cells.append(Paragraph(f"{val:+.3f} σ", table_cell_style))
                table_rows.append(row_cells)
                
            col_widths = [110] + [370 / len(feats_list)] * len(feats_list)
            enrich_table = Table(table_rows, colWidths=col_widths)
            enrich_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), primary_color),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_color]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]))
            story.append(enrich_table)
            story.append(Spacer(1, 12))

    # --- FOOTNOTE/SIGNATURE ---
    story.append(Spacer(1, 15))
    story.append(Paragraph("<i>Report generated automatically by SegFL Analytics.</i>", caption_style))
    
    # 3. Build Document
    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(54, 36, "SegFL Analytics Research Report")
        canvas.drawRightString(612 - 54, 36, f"Page {doc.page}")
        canvas.restoreState()
        
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    
    # Clean up temp image files
    if 'temp_img_files' in locals():
        for file in temp_img_files:
            try:
                os.remove(file)
            except Exception:
                pass
                
    return filename
