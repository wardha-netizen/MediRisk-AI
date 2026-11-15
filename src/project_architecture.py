
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


DIAGRAMS = {
    "System Architecture": "system_architecture.md",
    "Data Flow Diagram": "data_flow_diagram.md",
    "Machine Learning Workflow": "machine_learning_workflow.md",
}

DIAGRAM_FOLDER = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "diagrams"
)


def load_diagram(filename):
    path = DIAGRAM_FOLDER / filename

    if not path.is_file():
        return None

    content = path.read_text(encoding="utf-8")

    if "```mermaid" in content:
        return content.split("```mermaid", 1)[1].split(
            "```", 1
        )[0].strip()

    return content.strip()


def show_project_architecture():
    st.title("Project Architecture")

    st.write(
        "Explore how MediRisk AI processes datasets, "
        "trains Decision Tree classifiers and "
        "generates predictions."
    )

    selected = st.selectbox(
        "Select a diagram",
        list(DIAGRAMS.keys()),
    )

    descriptions = {
        "System Architecture": (
            "Shows how the Streamlit interface, "
            "preprocessing modules, disease models "
            "and analysis dashboards connect."
        ),
        "Data Flow Diagram": (
            "Illustrates how user inputs move through "
            "validation, preprocessing, prediction "
            "and result explanations."
        ),
        "Machine Learning Workflow": (
            "Shows dataset preparation, model training, "
            "testing and performance evaluation."
        ),
    }

    st.info(descriptions[selected])

    diagram = load_diagram(DIAGRAMS[selected])

    if not diagram:
        st.error("The selected diagram file was not found.")
        return

    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {
          margin: 0;
          padding: 16px;
          background: #ffffff;
          font-family: Arial, sans-serif;
        }
        .mermaid {
          display: flex;
          justify-content: center;
        }
        svg {
          max-width: 100%;
          height: auto;
        }
      </style>
    </head>
    <body>
      <pre class="mermaid">DIAGRAM_CONTENT</pre>
      <script type="module">
        import mermaid from
          'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';

        mermaid.initialize({
          startOnLoad: true,
          theme: 'default',
          securityLevel: 'strict'
        });
      </script>
    </body>
    </html>
    """

    html = html.replace(
        "DIAGRAM_CONTENT",
        diagram.replace("&", "&amp;").replace(
            "<", "&lt;"
        ).replace(">", "&gt;"),
    )

    components.html(
        html,
        height=750,
        scrolling=True,
    )

    st.caption(
        "These diagrams describe the academic "
        "software design and machine learning workflow."
    )
