
import streamlit as st


def go_to_dashboard():
    """Navigate from the welcome screen to the dashboard."""
    st.session_state["navigation"] = "Dashboard"


def render_welcome():
    """Display the MediRisk AI welcome screen."""

    st.write("")

    # Center the welcome content
    left, center, right = st.columns([1, 3, 1])

    with center:

        # Main welcome card
        with st.container(border=True):

            st.write("")
            st.write("")

            # App icon
            st.title("🩺")

            # App name and tagline
            st.title("MediRisk AI")

            st.subheader(
                "Intelligent Disease Risk Assessment"
            )

            st.write("")

            st.markdown(
                """
                Welcome to **MediRisk AI**!

                An interactive machine learning platform designed
                to assess risk across five major diseases.

                Explore AI-powered predictions, understand model
                decisions, examine datasets, and discover how
                machine learning supports early risk assessment.
                """
            )

            st.write("")
            st.divider()

            # Project disclaimer
            st.caption(
                "Educational project | Not a substitute for "
                "professional medical diagnosis."
            )

            st.write("")

            # Get Started button
            st.button(
                "Get Started →",
                type="primary",
                use_container_width=True,
                on_click=go_to_dashboard
            )

            st.write("")
