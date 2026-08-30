"""Golden evaluation dataset for the evidence-retrieval (RAG) layer.

Five synthetic (resume, JD) pairs across different engineering domains
(backend, ML, frontend, mobile, DevOps), each resume containing one
genuinely relevant item and one deliberate decoy per section (experience,
projects) so Recall@K is non-trivial to compute (not just 1-of-1). Two
negative-control queries pair a resume against an unrelated JD, where the
correct behaviour is to retrieve nothing -- used to check the retriever
doesn't fabricate relevance where none exists.

Relevance is hand-labelled via a predicate over each chunk's *reconstructed*
text (what `pipeline/retrieval.py::build_chunks` produces), not the resume
model itself, so labels stay correct even if chunk-text formatting changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from multi_agent_resume_screener.state import (
    EducationItem,
    ExperienceItem,
    ProjectItem,
    Section,
    StructuredJD,
    StructuredResume,
)


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    resume: StructuredResume
    jd: StructuredJD
    section: Section
    # True for chunk text that should count as relevant evidence for this
    # query. A query where no chunk in `resume` satisfies this is a
    # negative control (expected retrieval: nothing).
    relevant: Callable[[str], bool]


# --------------------------------------------------------------------------- #
# Resumes (one relevant + one decoy item per section, by design)
# --------------------------------------------------------------------------- #
BACKEND_RESUME = StructuredResume(
    name="Priya Sharma",
    skills=["Python", "FastAPI", "PostgreSQL", "Git", "Docker", "REST APIs", "HTML", "CSS"],
    experience=[
        ExperienceItem(
            company="PayEase Fintech",
            role="Backend Developer Intern",
            dates="2024",
            bullets=[
                "Built REST APIs in Python using FastAPI for a payments platform",
                "Designed PostgreSQL schemas and optimized slow queries",
            ],
        ),
        ExperienceItem(
            company="TechFest College Club",
            role="Campus Ambassador",
            dates="2023",
            bullets=["Promoted college tech fest on social media", "Coordinated event logistics"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Expense Tracker API",
            description="A FastAPI backend service with PostgreSQL storage for tracking personal expenses",
            tech=["FastAPI", "PostgreSQL", "Docker"],
        ),
        ProjectItem(
            name="College Cultural Fest Website",
            description="A static HTML/CSS landing page for the college's annual cultural fest",
            tech=["HTML", "CSS"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Computer Science", institute="NIT Trichy", year="2024")],
)

ML_RESUME = StructuredResume(
    name="Rahul Verma",
    skills=["Python", "PyTorch", "scikit-learn", "Pandas", "NumPy", "SQL", "Git"],
    experience=[
        ExperienceItem(
            company="VisionAI Labs",
            role="Machine Learning Intern",
            dates="2024",
            bullets=[
                "Trained CNN models in PyTorch for image classification",
                "Built data preprocessing pipelines with Pandas and NumPy",
            ],
        ),
        ExperienceItem(
            company="Retail Mart",
            role="Sales Associate",
            dates="2022-2023",
            bullets=["Assisted customers on the sales floor", "Managed inventory stock counts"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Sentiment Analysis on Twitter Data",
            description="An NLP pipeline using scikit-learn and TF-IDF to classify tweet sentiment",
            tech=["scikit-learn", "Python", "NLP"],
        ),
        ProjectItem(
            name="Personal Portfolio Website",
            description="A static personal portfolio site built with HTML, CSS, and vanilla JS",
            tech=["HTML", "CSS", "JavaScript"],
        ),
    ],
    education=[EducationItem(degree="B.Sc Statistics", institute="Delhi University", year="2023")],
)

FRONTEND_RESUME = StructuredResume(
    name="Ananya Iyer",
    skills=["JavaScript", "React", "Redux", "HTML", "CSS", "Tailwind CSS", "Git"],
    experience=[
        ExperienceItem(
            company="ShopEasy",
            role="Frontend Developer Intern",
            dates="2024",
            bullets=[
                "Built responsive UI components in React",
                "Managed application state with Redux and integrated REST APIs",
            ],
        ),
        ExperienceItem(
            company="Local NGO",
            role="Volunteer Teacher",
            dates="2023",
            bullets=["Taught basic computer literacy to schoolchildren"],
        ),
    ],
    projects=[
        ProjectItem(
            name="E-commerce Storefront UI",
            description="A React and Tailwind CSS storefront connected to a mock REST API",
            tech=["React", "Tailwind CSS", "Redux"],
        ),
        ProjectItem(
            name="Weather CLI Tool",
            description="A Python command-line script that fetches and prints weather forecasts",
            tech=["Python"],
        ),
    ],
    education=[EducationItem(degree="B.E. Information Technology", institute="Anna University", year="2024")],
)

MOBILE_RESUME = StructuredResume(
    name="Karan Mehta",
    skills=["Kotlin", "Android SDK", "Jetpack Compose", "Retrofit", "Room", "Git"],
    experience=[
        ExperienceItem(
            company="AppForge Studios",
            role="Android Developer Intern",
            dates="2024",
            bullets=[
                "Built Android app features in Kotlin using Jetpack Compose",
                "Integrated Retrofit for REST networking and Room for local storage",
            ],
        ),
        ExperienceItem(
            company="Freelance",
            role="Content Writer",
            dates="2022-2023",
            bullets=["Wrote blog articles and optimized them for SEO"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Habit Tracker Android App",
            description="A Kotlin Android app using Jetpack Compose and Room for local persistence",
            tech=["Kotlin", "Jetpack Compose", "Room"],
        ),
        ProjectItem(
            name="Excel Automation Macro",
            description="A VBA macro that automates monthly expense report formatting in Excel",
            tech=["VBA"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Electronics and Communication", institute="VIT Vellore", year="2023")],
)

DEVOPS_RESUME = StructuredResume(
    name="Sneha Nair",
    skills=["Docker", "Kubernetes", "AWS", "GitHub Actions", "Terraform", "Linux", "Git"],
    experience=[
        ExperienceItem(
            company="CloudNine Systems",
            role="DevOps Intern",
            dates="2024",
            bullets=[
                "Set up CI/CD pipelines with GitHub Actions and Docker",
                "Deployed and managed containerized services on AWS EC2",
            ],
        ),
        ExperienceItem(
            company="City Library",
            role="Front Desk Assistant",
            dates="2022",
            bullets=["Assisted visitors and managed book checkouts"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Kubernetes Cluster Monitoring Dashboard",
            description="A self-hosted Kubernetes cluster monitored with Prometheus and Grafana",
            tech=["Kubernetes", "Prometheus", "Grafana"],
        ),
        ProjectItem(
            name="Recipe Sharing Mobile App",
            description="A Flutter mobile app for sharing and browsing recipes",
            tech=["Flutter", "Dart"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Computer Science", institute="IIIT Hyderabad", year="2022")],
)


# --------------------------------------------------------------------------- #
# Job descriptions
# --------------------------------------------------------------------------- #
JD_BACKEND = StructuredJD(
    title="Backend Engineer",
    required_skills=["Python", "FastAPI", "PostgreSQL", "REST APIs"],
    nice_to_have_skills=["Docker"],
    responsibilities=["Design and build REST APIs", "Work with PostgreSQL databases"],
    qualifications=["Bachelor's degree in Computer Science or a related field"],
)
JD_ML = StructuredJD(
    title="Machine Learning Engineer",
    required_skills=["Python", "PyTorch", "scikit-learn", "Pandas"],
    nice_to_have_skills=["NLP"],
    responsibilities=["Train and evaluate machine learning models", "Build data preprocessing pipelines"],
    qualifications=["Degree in Statistics, Mathematics, Computer Science, or a related field"],
)
JD_FRONTEND = StructuredJD(
    title="Frontend Engineer",
    required_skills=["JavaScript", "React", "HTML", "CSS"],
    nice_to_have_skills=["Redux", "Tailwind CSS"],
    responsibilities=["Build responsive user interfaces in React", "Integrate the frontend with REST APIs"],
)
JD_MOBILE = StructuredJD(
    title="Android Engineer",
    required_skills=["Kotlin", "Android SDK", "Jetpack Compose"],
    nice_to_have_skills=["Retrofit", "Room"],
    responsibilities=["Develop native Android features in Kotlin", "Work with local databases and REST clients"],
)
JD_DEVOPS = StructuredJD(
    title="DevOps Engineer",
    required_skills=["Docker", "Kubernetes", "AWS"],
    nice_to_have_skills=["Terraform", "GitHub Actions"],
    responsibilities=["Build and maintain CI/CD pipelines", "Manage containerized deployments on AWS"],
)


# --------------------------------------------------------------------------- #
# Golden queries: (resume, matched JD, section) -> relevance predicate
# --------------------------------------------------------------------------- #
def _contains_any(*keywords: str) -> Callable[[str], bool]:
    return lambda text: any(kw.lower() in text.lower() for kw in keywords)


GOLDEN_QUERIES: list[GoldenQuery] = [
    # --- Backend / Priya ---
    GoldenQuery("backend_skills", BACKEND_RESUME, JD_BACKEND, "skills", _contains_any("FastAPI", "PostgreSQL")),
    GoldenQuery("backend_experience", BACKEND_RESUME, JD_BACKEND, "experience", _contains_any("FastAPI", "PostgreSQL")),
    GoldenQuery("backend_projects", BACKEND_RESUME, JD_BACKEND, "projects", _contains_any("FastAPI", "PostgreSQL")),
    GoldenQuery("backend_education", BACKEND_RESUME, JD_BACKEND, "education", _contains_any("Computer Science")),
    # --- ML / Rahul ---
    GoldenQuery("ml_skills", ML_RESUME, JD_ML, "skills", _contains_any("PyTorch", "scikit-learn", "Pandas")),
    GoldenQuery("ml_experience", ML_RESUME, JD_ML, "experience", _contains_any("PyTorch", "CNN", "Pandas")),
    GoldenQuery("ml_projects", ML_RESUME, JD_ML, "projects", _contains_any("scikit-learn", "NLP", "TF-IDF")),
    # --- Frontend / Ananya ---
    GoldenQuery("frontend_skills", FRONTEND_RESUME, JD_FRONTEND, "skills", _contains_any("React", "Redux")),
    GoldenQuery("frontend_experience", FRONTEND_RESUME, JD_FRONTEND, "experience", _contains_any("React", "Redux")),
    GoldenQuery("frontend_projects", FRONTEND_RESUME, JD_FRONTEND, "projects", _contains_any("React", "Tailwind")),
    # --- Mobile / Karan ---
    GoldenQuery("mobile_skills", MOBILE_RESUME, JD_MOBILE, "skills", _contains_any("Kotlin", "Jetpack Compose")),
    GoldenQuery("mobile_experience", MOBILE_RESUME, JD_MOBILE, "experience", _contains_any("Kotlin", "Jetpack Compose", "Retrofit")),
    GoldenQuery("mobile_projects", MOBILE_RESUME, JD_MOBILE, "projects", _contains_any("Kotlin", "Jetpack Compose", "Room")),
    # --- DevOps / Sneha ---
    GoldenQuery("devops_skills", DEVOPS_RESUME, JD_DEVOPS, "skills", _contains_any("Docker", "Kubernetes", "AWS")),
    GoldenQuery("devops_experience", DEVOPS_RESUME, JD_DEVOPS, "experience", _contains_any("CI/CD", "GitHub Actions", "AWS EC2")),
    GoldenQuery("devops_projects", DEVOPS_RESUME, JD_DEVOPS, "projects", _contains_any("Kubernetes", "Prometheus", "Grafana")),
]

# Negative controls: resume x unrelated JD. Nothing in these resumes should
# be relevant to these JDs' requirements -- correct retrieval is empty.
NEGATIVE_CONTROLS: list[GoldenQuery] = [
    GoldenQuery("backend_resume_vs_ml_jd_skills", BACKEND_RESUME, JD_ML, "skills", lambda text: False),
    GoldenQuery("backend_resume_vs_ml_jd_experience", BACKEND_RESUME, JD_ML, "experience", lambda text: False),
    GoldenQuery("frontend_resume_vs_devops_jd_skills", FRONTEND_RESUME, JD_DEVOPS, "skills", lambda text: False),
    GoldenQuery("frontend_resume_vs_devops_jd_experience", FRONTEND_RESUME, JD_DEVOPS, "experience", lambda text: False),
]
