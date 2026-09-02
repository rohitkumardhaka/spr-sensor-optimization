from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def create_text_only_spr_report():
    filename = "SPR_Optimizer_Detailed_Report_TextOnly.pdf"
    document = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # --- Custom Styles ---
    title_style = styles['Title']
    heading_style = styles['Heading2']
    subheading_style = styles['Heading3']
    body_style = styles['BodyText']
    body_style.spaceAfter = 12  # Add spacing after paragraphs

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['BodyText'],
        bulletIndent=10,
        leftIndent=20,
        spaceAfter=5
    )

    # --- Title Page ---
    story.append(
        Paragraph("SPR Optimizer: Comprehensive Project Report", title_style))
    story.append(Spacer(1, 20))

    intro_text = """
    This document provides a detailed technical analysis of the Surface Plasmon Resonance (SPR) Optimizer project. 
    It covers the theoretical framework, computational methodology, optimization results, and a critical review of the codebase.
    """
    story.append(Paragraph(intro_text, body_style))
    story.append(Spacer(1, 20))

    # --- Section 1: Project Description ---
    story.append(Paragraph("1. Project Description", heading_style))

    desc_text = """
    The SPR Optimizer is a computational tool designed to automate the design of high-performance optical biosensors. 
    The primary objective is to maximize the <b>Figure of Merit (FOM)</b>, a metric that balances high sensitivity 
    (a large shift in resonance angle per unit of refractive index change) with a narrow resonance linewidth (Full Width at Half Maximum).
    """
    story.append(Paragraph(desc_text, body_style))

    desc_text_2 = """
    The system replaces traditional trial-and-error engineering with <b>Bayesian Optimization</b>. By simulating the 
    physical interactions of light with multilayer structures, the software iteratively learns which material combinations 
    and layer thicknesses yield the optimal sensor response.
    """
    story.append(Paragraph(desc_text_2, body_style))

    # --- Section 2: Methodology ---
    story.append(Paragraph("2. Methodology & Physics", heading_style))

    story.append(Paragraph(
        "<b>The Physics Engine (Transfer Matrix Method):</b>", subheading_style))
    physics_text = """
    The core simulation engine uses the Transfer Matrix Method (TMM) to solve Maxwell's equations for a planar multilayer stack. 
    For every candidate design, the system calculates the Fresnel reflection coefficients at each interface and cascades them 
    to determine the total optical reflectance across a range of incident angles (typically 20° to 89°).
    """
    story.append(Paragraph(physics_text, body_style))

    story.append(
        Paragraph("<b>The Optimization Algorithm (Optuna):</b>", subheading_style))
    opt_text = """
    The project utilizes the Optuna library to perform Bayesian hyperparameter optimization. The search space consists of four key parameters:
    """
    story.append(Paragraph(opt_text, body_style))

    param_items = [
        ListItem(Paragraph(
            "<b>Metal Choice:</b> Selecting between Silver (Ag), Gold (Au), or Copper (Cu).", bullet_style)),
        ListItem(Paragraph(
            "<b>D2 (Metal Thickness):</b> Varied between 20 nm and 120 nm.", bullet_style)),
        ListItem(Paragraph(
            "<b>D3 (Dielectric Layer 1):</b> Varied between 0.5 nm and 20 nm.", bullet_style)),
        ListItem(Paragraph(
            "<b>D4 (Dielectric Layer 2):</b> Varied between 0.1 nm and 2 nm.", bullet_style))
    ]
    story.append(ListFlowable(param_items, bulletType='bullet'))

    # --- Section 3: Detailed Results Analysis ---
    story.append(Paragraph("3. Results Analysis", heading_style))

    story.append(
        Paragraph("<b>Optimization Convergence:</b>", subheading_style))
    conv_text = """
    Analysis of the optimization history reveals a rapid convergence profile. The algorithm identified a near-optimal 
    solution (FOM ~187) within the first 50 trials. The remaining ~19,950 trials were spent exploring the parameter space 
    but failed to find a superior configuration. This suggests the solution space has a distinct, steep global maximum 
    that is easily discoverable.
    """
    story.append(Paragraph(conv_text, body_style))

    story.append(Paragraph("<b>Parameter Sensitivity:</b>", subheading_style))
    sens_text = """
    A sensitivity analysis quantifying the impact of each variable on the final FOM score yielded the following hierarchy of importance:
    """
    story.append(Paragraph(sens_text, body_style))

    importance_items = [
        ListItem(Paragraph("<b>1. Dielectric Layer 1 (D3) - 43% Importance:</b> The thickness of the PbMoO4 layer is the single most critical factor. This layer likely acts to tune the propagation constant of the surface plasmon to match the prism coupling.", bullet_style)),
        ListItem(Paragraph("<b>2. Metal Thickness (D2) - 30% Importance:</b> The thickness of the metal layer is crucial for determining the coupling efficiency and radiative damping of the plasmon.", bullet_style)),
        ListItem(Paragraph("<b>3. Metal Choice - 13% Importance:</b> Surprisingly, the specific material (Ag vs Au) was less critical than the geometry of the layers.", bullet_style))
    ]
    story.append(ListFlowable(importance_items, bulletType='bullet'))

    story.append(
        Paragraph("<b>Optimal Configuration Found:</b>", subheading_style))
    config_text = """
    The highest performing design identified by the system was a <b>Gold (Au)</b> based sensor. 
    Specific parameters included a metal thickness of approximately <b>24.4 nm</b> and a primary dielectric layer of <b>13.5 nm</b>.
    """
    story.append(Paragraph(config_text, body_style))

    # --- Section 4: Technical Critique ---
    story.append(Paragraph("4. Technical Critique & Issues", heading_style))

    critique_text = """
    While the results are promising, a code review identified three specific issues that may affect the validity of the findings:
    """
    story.append(Paragraph(critique_text, body_style))

    bug_items = [
        ListItem(Paragraph("<b>Critical Bug (Edge Artifacts):</b> The FOM calculation uses `argmin` to find the resonance dip. If the dip shifts outside the simulated angular range, the algorithm selects the boundary index, resulting in a calculation of 'infinite' or artificially high sensitivity. This likely explains the unusually high FOM of 187 for Gold.", bullet_style)),
        ListItem(Paragraph("<b>Objective Discontinuity:</b> The optimization loop switches between 'coarse' and 'fine' angular resolution based on a threshold score. This creates a step-function in the optimization landscape, which can confuse the Bayesian solver.", bullet_style)),
        ListItem(Paragraph("<b>Numerical Instability:</b> The Fresnel equations used in the physics engine are susceptible to division-by-zero errors at the critical angle, which can lead to crashes or NaN values during simulation.", bullet_style))
    ]
    story.append(ListFlowable(bug_items, bulletType='bullet'))

    # Build PDF
    document.build(story)
    print(f"PDF generated successfully: {filename}")


if __name__ == "__main__":
    create_text_only_spr_report()
