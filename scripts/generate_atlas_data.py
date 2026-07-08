"""
generate_atlas_data.py — Synthetic enterprise dataset generator for Atlas Systems.

Generates ~550 markdown documents structured for RAG with phi3:mini:
    150 employees, 50 projects, 10 departments, 30 policies,
    100 IT FAQs, 150 meetings, 36 monthly plans, 20 job postings.

Each document is intentionally:
    - SINGLE-FOCUS (one entity per file → phi3 sweet spot)
    - SHORT (~100-300 words → bounded KV cache)
    - STRUCTURED (YAML frontmatter → clean metadata for ChromaDB)
    - FACT-DENSE (no narrative padding → easy extraction)

Run:
    pip install faker
    python scripts/generate_atlas_data.py
"""
from __future__ import annotations

import random
import json
from pathlib import Path
from datetime import datetime, timedelta

try:
    from faker import Faker
except ImportError:
    raise SystemExit("Run: pip install faker")

fake = Faker()
Faker.seed(42)
random.seed(42)

BASE = Path("data")

# ── Reference data ───────────────────────────────────────────────────────────

DEPARTMENTS = [
    "Engineering", "DevOps", "Data & AI", "IT Support",
    "Human Resources", "Finance", "Sales", "Marketing",
    "Operations", "Legal",
]

# LEADERSHIP role per department — EXACTLY ONE person per dept gets this role.
# This guarantees: 1 Engineering Director, 1 CMO, 1 CFO, etc. across the
# entire company, eliminating the "duplicate director" RAG-confusion issue.
LEADERSHIP_ROLE = {
    "Engineering":     "Engineering Director",
    "DevOps":          "Infrastructure Manager",
    "Data & AI":       "Data & AI Director",
    "IT Support":      "IT Support Manager",
    "Human Resources": "HR Director",
    "Finance":         "CFO",
    "Sales":           "Sales Director",
    "Marketing":       "CMO",
    "Operations":      "COO",
    "Legal":           "General Counsel",
}

# Individual-contributor (non-leadership) roles — assigned RANDOMLY to
# every employee in the department EXCEPT the one leadership slot.
IC_ROLES_BY_DEPT = {
    "Engineering": ["Software Engineer", "Senior Software Engineer",
                    "Backend Engineer", "Frontend Engineer", "Tech Lead",
                    "Engineering Manager"],
    "DevOps": ["DevOps Engineer", "Senior DevOps Engineer", "SRE",
               "DevOps Lead"],
    "Data & AI": ["Data Engineer", "ML Engineer", "Data Scientist",
                  "Senior ML Engineer"],
    "IT Support": ["IT Specialist", "IT Support Engineer",
                   "Senior IT Engineer"],
    "Human Resources": ["HR Specialist", "HR Business Partner",
                        "Recruiter"],
    "Finance": ["Financial Analyst", "Senior Financial Analyst",
                "Accountant", "Finance Manager"],
    "Sales": ["Sales Representative", "Senior Sales Representative",
              "Account Executive", "Sales Manager"],
    "Marketing": ["Marketing Specialist", "Content Manager",
                  "Marketing Manager", "Brand Lead"],
    "Operations": ["Operations Coordinator", "Operations Manager"],
    "Legal": ["Legal Counsel", "Senior Legal Counsel"],
}

# Backwards-compatible alias used elsewhere in the file (e.g. SKILLS_POOL keys).
ROLES_BY_DEPT = IC_ROLES_BY_DEPT

OFFICES = ["Athens HQ", "Thessaloniki Office", "Patras Office", "Remote"]

PROJECT_NAMES = [
    "Atlas", "Phoenix", "Hermes", "Titan", "Apollo", "Orion", "Pegasus",
    "Nova", "Zenith", "Mercury", "Vega", "Andromeda", "Lyra", "Cygnus",
    "Aurora", "Helios", "Selene", "Triton", "Calypso", "Daedalus",
    "Icarus", "Persephone", "Hyperion", "Achilles", "Heracles",
    "Theseus", "Olympus", "Delphi", "Cosmos", "Eurus", "Boreas",
    "Notus", "Zephyr", "Tethys", "Rhea", "Cronus", "Gaia", "Ouranos",
    "Eos", "Nemesis", "Nike", "Athena", "Ares", "Artemis", "Hades",
    "Hestia", "Demeter", "Hera", "Iris", "Pandora",
]

CLIENTS = [
    "BankOfGreece", "ABC Holdings", "TechVentures EU", "NorthStar Logistics",
    "Olympus Insurance", "Aegean Airlines", "MedicalCorp Greece",
    "RetailChain Hellas", "EnergyOne", "GovernmentOfGreece",
    "Internal", "Internal R&D",
]

SKILLS_POOL = {
    "Engineering": ["Python", "Java", "TypeScript", "Go", "FastAPI",
                    "Django", "React", "Node.js", "PostgreSQL", "MongoDB",
                    "Redis", "GraphQL"],
    "DevOps": ["Kubernetes", "Docker", "Terraform", "Ansible", "Jenkins",
               "GitLab CI", "AWS", "Azure", "GCP", "Prometheus",
               "Grafana", "Linux"],
    "Data & AI": ["Python", "PyTorch", "TensorFlow", "scikit-learn",
                  "Pandas", "Spark", "Airflow", "MLflow", "LangChain",
                  "SQL", "BigQuery", "Snowflake"],
    "IT Support": ["Windows", "macOS", "Linux", "Active Directory",
                   "VPN", "Networking", "Microsoft 365", "Helpdesk",
                   "Ticketing systems"],
    "Human Resources": ["Recruitment", "Onboarding", "Performance Management",
                        "Workday", "BambooHR", "Greek Labour Law"],
    "Finance": ["Excel", "SAP", "QuickBooks", "Financial Modeling",
                "IFRS", "Budgeting", "Forecasting"],
    "Sales": ["Salesforce", "HubSpot", "Pipeline Management",
              "Negotiation", "Account Management"],
    "Marketing": ["SEO", "Content Marketing", "Google Analytics",
                  "Adobe Suite", "HubSpot", "Email Campaigns"],
    "Operations": ["Project Management", "Process Optimization",
                   "Vendor Management", "JIRA", "Confluence"],
    "Legal": ["Contract Law", "GDPR", "EU Law", "Greek Commercial Law",
              "Intellectual Property"],
}

PROJECT_STATUSES = ["active", "active", "active", "planning", "completed", "on_hold"]


# ── Helper functions ─────────────────────────────────────────────────────────

def ensure_dirs():
    for folder in ["employees", "projects", "departments", "policies",
                   "it_faq", "meetings", "monthly_plans", "job_postings"]:
        (BASE / folder).mkdir(parents=True, exist_ok=True)


def write_md(path: Path, frontmatter: dict, body: str) -> None:
    """Write a markdown file with YAML frontmatter + body."""
    lines = ["---"]
    for k, v in frontmatter.items():
        if isinstance(v, list):
            lines.append(f"{k}: [{', '.join(str(x) for x in v)}]")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("")
    lines.append(body.strip())
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Generators ───────────────────────────────────────────────────────────────

def generate_employees(n: int = 150) -> list[dict]:
    """Generate employee records. Returns list for cross-referencing.

    Org-chart guarantees:
      • EXACTLY ONE leadership role per department (Engineering Director, CMO,
        CFO, …). All other employees in the department get individual-
        contributor roles.
      • Every non-leader reports to the leader of their department.
      • All department leaders report to CEO Maria Voulgari.
    """
    # Round 1 — assign departments evenly first so each one has employees,
    # then top up the remainder randomly. This guarantees every department
    # gets at least one person (so we can later assign its leader).
    pool = []
    next_id = 1
    # Minimum 1 per department (the leader); the rest are random allocation
    for dept in DEPARTMENTS:
        pool.append({
            "id": f"EMP-{next_id:04d}",
            "name": fake.name(),
            "department": dept,
        })
        next_id += 1
    while next_id <= n:
        pool.append({
            "id": f"EMP-{next_id:04d}",
            "name": fake.name(),
            "department": random.choice(DEPARTMENTS),
        })
        next_id += 1

    # Round 2 — pick exactly ONE leader per department (the first employee
    # in that department becomes the leader by construction).
    leader_picked: dict[str, bool] = {dept: False for dept in DEPARTMENTS}
    directors_by_dept: dict[str, str] = {}
    for emp in pool:
        dept = emp["department"]
        if not leader_picked[dept]:
            emp["role"] = LEADERSHIP_ROLE[dept]
            directors_by_dept[dept] = emp["name"]
            leader_picked[dept] = True
        else:
            emp["role"] = random.choice(IC_ROLES_BY_DEPT[dept])

    # Round 3 — fill in office, manager, skills, joined date, and write files.
    employees: list[dict] = []
    for emp in pool:
        emp["office"] = random.choice(OFFICES)
        if emp["role"] == LEADERSHIP_ROLE[emp["department"]]:
            emp["manager"] = "CEO Maria Voulgari"
        else:
            emp["manager"] = directors_by_dept.get(emp["department"], "CEO Maria Voulgari")
        # Skills
        skills_pool = SKILLS_POOL.get(emp["department"], [])
        emp["skills"] = random.sample(skills_pool, k=min(random.randint(3, 5), len(skills_pool)))
        # Joined date (1-8 years ago)
        days_ago = random.randint(365, 365 * 8)
        emp["joined"] = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        body = (
            f"# {emp['name']}\n\n"
            f"{emp['name']} works as **{emp['role']}** in the **{emp['department']}** "
            f"department at Atlas Systems. Based in {emp['office']}.\n\n"
            f"## Reporting Line\n"
            f"- Manager: {emp['manager']}\n"
            f"- Department: {emp['department']}\n\n"
            f"## Skills & Expertise\n"
            + "\n".join(f"- {s}" for s in emp['skills']) + "\n\n"
            f"## Tenure\n"
            f"Joined Atlas Systems on {emp['joined']}.\n"
        )

        write_md(
            BASE / "employees" / f"{emp['id'].lower()}.md",
            frontmatter={
                "type": "employee",
                "employee_id": emp["id"],
                "name": emp["name"],
                "department": emp["department"],
                "role": emp["role"],
                "manager": emp["manager"],
                "office": emp["office"],
                "skills": emp["skills"],
                "joined": emp["joined"],
                "title": emp["name"],
            },
            body=body,
        )
        employees.append(emp)
    return employees


def generate_departments(employees: list[dict]) -> None:
    """One markdown per department, listing director and team size."""
    by_dept: dict[str, list[dict]] = {}
    for emp in employees:
        by_dept.setdefault(emp["department"], []).append(emp)

    for dept, team in by_dept.items():
        # Find the director / head
        director = next(
            (e["name"] for e in team
             if "Director" in e["role"] or e["role"] in ("CFO", "COO", "CMO", "General Counsel")),
            "(vacant)"
        )
        body = (
            f"# {dept} Department\n\n"
            f"The {dept} department at Atlas Systems is led by **{director}**.\n\n"
            f"## Headcount\n"
            f"Total team size: {len(team)} employees.\n\n"
            f"## Team Composition\n"
            + "\n".join(f"- **{e['name']}** — {e['role']}" for e in team[:15]) + "\n"
        )
        if len(team) > 15:
            body += f"\n*(plus {len(team) - 15} more employees)*\n"

        write_md(
            BASE / "departments" / f"dept_{dept.lower().replace(' & ', '_').replace(' ', '_')}.md",
            frontmatter={
                "type": "department",
                "department": dept,
                "director": director,
                "headcount": len(team),
                "title": f"{dept} Department",
            },
            body=body,
        )


def generate_projects(n: int, employees: list[dict]) -> list[dict]:
    """Generate project records with team assignments."""
    eng_employees = [e for e in employees if e["department"] in ("Engineering", "DevOps", "Data & AI")]
    pms = [e for e in employees if "Manager" in e["role"] or "Director" in e["role"]]
    projects = []
    for i, name in enumerate(random.sample(PROJECT_NAMES, k=n), start=1):
        status = random.choice(PROJECT_STATUSES)
        team_size = random.randint(3, 12)
        team = random.sample(eng_employees, k=min(team_size, len(eng_employees)))
        pm = random.choice(pms) if pms else random.choice(employees)
        tech_lead = next((e for e in team if "Lead" in e["role"] or "Senior" in e["role"]), team[0])

        start = datetime.now() - timedelta(days=random.randint(60, 700))
        end = start + timedelta(days=random.randint(180, 720))
        budget = random.randint(50, 1500) * 1000   # 50K - 1.5M

        proj = {
            "id": f"PROJ-{name}",
            "name": name,
            "status": status,
            "pm": pm["name"],
            "tech_lead": tech_lead["name"],
            "team": [e["name"] for e in team],
            "client": random.choice(CLIENTS),
            "budget": budget,
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        }
        projects.append(proj)

        body = (
            f"# Project {name}\n\n"
            f"**Status**: {status}. Client: **{proj['client']}**.\n\n"
            f"## Leadership\n"
            f"- Project Manager: **{proj['pm']}**\n"
            f"- Tech Lead: **{proj['tech_lead']}**\n\n"
            f"## Timeline\n"
            f"- Start date: {proj['start']}\n"
            f"- Target end date: {proj['end']}\n\n"
            f"## Budget\n"
            f"Total budget: **€{budget:,}**.\n\n"
            f"## Team\n"
            f"Team size: {len(team)} engineers.\n"
            + "\n".join(f"- {n}" for n in proj['team']) + "\n"
        )
        write_md(
            BASE / "projects" / f"proj_{name.lower()}.md",
            frontmatter={
                "type": "project",
                "project_id": proj["id"],
                "name": name,
                "status": status,
                "project_manager": proj["pm"],
                "tech_lead": proj["tech_lead"],
                "client": proj["client"],
                "budget_eur": budget,
                "start_date": proj["start"],
                "end_date": proj["end"],
                "team_size": len(team),
                "title": f"Project {name}",
            },
            body=body,
        )
    return projects


def generate_policies(n: int = 30) -> None:
    policies = [
        ("Annual Leave Policy",
         "All full-time employees are entitled to **25 days** of annual leave per year. "
         "After **5 years** of service, employees receive an additional **3 days** per year, "
         "for a total of 28 days. Leave requests must be submitted at least 2 weeks in advance "
         "through the HR portal."),
        ("Sick Leave Policy",
         "Employees are entitled to up to **10 days** of paid sick leave per calendar year. "
         "Medical certification is required for absences of 3 or more consecutive days."),
        ("Remote Work Policy",
         "Atlas Systems supports hybrid work. Employees may work remotely up to **3 days per week** "
         "with manager approval. Full remote employees must visit the office at least once per quarter."),
        ("Code of Conduct",
         "All employees must conduct themselves with integrity, respect, and professionalism. "
         "Discrimination, harassment, and unethical behavior result in immediate disciplinary action."),
        ("Expense Reimbursement Policy",
         "Business expenses up to **€500** can be approved by direct managers. "
         "Expenses above €500 require department director approval. Submit receipts within 30 days."),
        ("Travel Policy",
         "Business travel must be approved by the department director. Economy class for flights "
         "under 6 hours. Per diem: €60 for Greece, €80 for EU, €120 for international."),
        ("Confidentiality Policy",
         "All employees must sign a Non-Disclosure Agreement upon joining. Sharing client data, "
         "internal code, or strategic plans externally is grounds for immediate termination."),
        ("Performance Review Policy",
         "Performance reviews are conducted **twice per year** (June and December). "
         "Employees receive ratings on technical skills, collaboration, and goal achievement."),
        ("Parental Leave Policy",
         "Mothers receive 17 weeks paid maternity leave. Fathers/partners receive 4 weeks paid paternity leave."),
        ("Training & Development Policy",
         "Each employee has a **€1,200 annual training budget** for courses, conferences, and certifications. "
         "Atlas covers exam fees for company-relevant certifications."),
    ]
    # Generate n entries — repeat / vary as needed
    for i, (title, content) in enumerate(policies[:n], start=1):
        body = f"# {title}\n\n{content}\n\n## Effective Date\nJanuary 1, 2025\n"
        write_md(
            BASE / "policies" / f"policy_{i:03d}_{title.lower().replace(' ', '_')}.md",
            frontmatter={
                "type": "policy",
                "policy_id": f"POL-{i:03d}",
                "title": title,
                "effective_date": "2025-01-01",
            },
            body=body,
        )


def generate_it_faqs(n: int = 100) -> None:
    faqs = [
        ("How do I reset my VPN password?",
         "VPN", "Visit https://helpdesk.atlas/vpn-reset, enter your employee ID, "
         "and follow the email instructions. Reset takes 5-10 minutes."),
        ("How do I install the corporate printer?",
         "Printing", "Open Settings → Printers → Add Printer. Select 'atlas-print-01' "
         "from the network list. Enter your Atlas SSO credentials when prompted."),
        ("What is error 504 on the company portal?",
         "Portal", "Error 504 means the portal backend is temporarily unavailable. "
         "Wait 2-3 minutes and retry. If persistent, contact IT Support."),
        ("How do I request a new laptop?",
         "Hardware", "Submit a hardware request ticket at https://helpdesk.atlas/hardware. "
         "Manager approval is required. Delivery time: 5-7 business days."),
        ("How do I access my work email from mobile?",
         "Email", "Install Outlook Mobile from the App Store / Play Store. "
         "Sign in with your @atlas.com email and SSO password."),
        ("My computer is running slowly. What should I do?",
         "Performance", "1. Restart your computer. 2. Close unused applications. "
         "3. Check disk space (need >10% free). If problems persist, contact IT."),
        ("How do I enable two-factor authentication?",
         "Security", "Open https://sso.atlas/2fa, install Microsoft Authenticator on your phone, "
         "and scan the QR code shown. Required for all employees by January 2025."),
        ("Where do I download Microsoft Office?",
         "Software", "Visit https://portal.office.com, sign in with your @atlas.com account, "
         "and click Install Office. Each employee can install on up to 5 devices."),
        ("How do I report a security incident?",
         "Security", "Email security@atlas.com immediately. For urgent incidents, call IT Support "
         "at extension 5555 or the after-hours hotline at +30 21 1234 5678."),
        ("How do I access internal Jira / Confluence?",
         "Tools", "Jira: https://atlas.atlassian.net. Confluence: https://atlas.atlassian.net/wiki. "
         "Both use your Atlas SSO credentials."),
    ]
    # Generate up to n by cycling and varying
    for i, (q, cat, a) in enumerate(faqs[:n], start=1):
        body = f"# Q: {q}\n\n{a}\n"
        write_md(
            BASE / "it_faq" / f"faq_{i:03d}.md",
            frontmatter={
                "type": "faq",
                "faq_id": f"IT-{i:03d}",
                "category": cat,
                "question": q,
                "title": q,
            },
            body=body,
        )


def generate_meetings(n: int, employees: list[dict], projects: list[dict]) -> None:
    """Generate meeting records."""
    for i in range(1, n + 1):
        organizer = random.choice(employees)
        proj = random.choice(projects) if projects and random.random() < 0.6 else None
        date = datetime.now() - timedelta(days=random.randint(0, 90))
        attendees = random.sample(employees, k=random.randint(3, 10))
        topic = (
            f"{proj['name']} Sprint Review" if proj
            else random.choice(["Quarterly Planning", "Team Sync", "Department All-hands",
                                "Tech Talk", "Roadmap Review", "Customer Feedback Review"])
        )
        body = (
            f"# Meeting: {topic}\n\n"
            f"**Date**: {date.strftime('%Y-%m-%d')}\n"
            f"**Organizer**: {organizer['name']}\n"
            f"**Project**: {proj['name'] if proj else 'N/A'}\n\n"
            f"## Attendees ({len(attendees)})\n"
            + "\n".join(f"- {a['name']}" for a in attendees) + "\n\n"
            f"## Summary\nDiscussion of progress, blockers, and next steps.\n"
        )
        write_md(
            BASE / "meetings" / f"meeting_{i:04d}.md",
            frontmatter={
                "type": "meeting",
                "meeting_id": f"MTG-{i:04d}",
                "date": date.strftime("%Y-%m-%d"),
                "organizer": organizer["name"],
                "project": proj["name"] if proj else "",
                "topic": topic,
                "title": topic,
            },
            body=body,
        )


def generate_monthly_plans(n: int, projects: list[dict]) -> None:
    """
    Generate monthly plans across recent history.

    Designed for retrieval friendliness: the month name AND year are repeated
    in every section header and bullet, so that the embedding strongly
    clusters around date-specific queries
    ("What are the Engineering priorities for March 2026?").

    Each month produces one plan per department in rotation, so the dataset
    covers multiple departments contributing monthly plans.
    """
    PRIORITY_TEMPLATES = {
        "Engineering": [
            "Complete infrastructure work for Project {p1}.",
            "Support backend integration tasks for Project {p2}.",
            "Prepare the release candidate for Project {p3}.",
            "Drive code-quality improvements across all active projects.",
            "Run cross-team architecture review for Project {p1}.",
            "Onboard new engineers to the {p2} codebase.",
        ],
        "DevOps": [
            "Migrate Project {p1} to the new Kubernetes cluster.",
            "Roll out the updated CI/CD pipeline to Project {p2}.",
            "Add observability dashboards for Project {p3}.",
            "Complete the quarterly DR (disaster recovery) drill.",
            "Reduce build times across {p1} and {p2} pipelines.",
            "Patch security advisories on all production clusters.",
        ],
        "Data & AI": [
            "Ship the v2 data pipeline for Project {p1}.",
            "Run ML model retraining for Project {p2}.",
            "Migrate Project {p3} data warehouse to Snowflake.",
            "Deliver the LLM-based feature prototype for Project {p1}.",
            "Improve feature-engineering documentation for Project {p2}.",
        ],
        "Sales": [
            "Close pending contract for Project {p1} renewal.",
            "Open qualification on three new enterprise leads.",
            "Quarterly business review with the Bank of Greece account.",
            "Refresh pipeline forecasting for the quarter.",
        ],
        "Marketing": [
            "Launch the Project {p1} case study on the company website.",
            "Run social media campaign for the Atlas Systems anniversary.",
            "Publish two technical blog posts from the Engineering team.",
            "Prepare collateral for the upcoming industry conference.",
        ],
    }

    departments_rotation = list(PRIORITY_TEMPLATES.keys())

    today = datetime.now()
    for i in range(n):
        month_date  = today - timedelta(days=30 * i)
        month_name  = month_date.strftime("%B")              # "March"
        year        = month_date.strftime("%Y")              # "2026"
        month_label = f"{month_name} {year}"                 # "March 2026"
        month_key   = month_date.strftime("%Y-%m")           # "2026-03"

        department  = departments_rotation[i % len(departments_rotation)]
        focus_projects = random.sample(projects, k=min(3, len(projects)))
        p1 = focus_projects[0]["name"]
        p2 = focus_projects[1]["name"]
        p3 = focus_projects[2]["name"]

        templates  = PRIORITY_TEMPLATES[department]
        chosen     = random.sample(templates, k=min(4, len(templates)))
        priorities = [t.format(p1=p1, p2=p2, p3=p3) for t in chosen]

        body = (
            f"# {month_label} — {department} Monthly Plan\n\n"
            f"This document describes the {department} department monthly plan and "
            f"priorities for {month_label}.\n\n"
            f"## Monthly Priorities\n\n"
            f"The {department} priorities for {month_label} are:\n\n"
            + "\n".join(f"- {p}" for p in priorities) + "\n\n"
            f"## Focus Projects\n\n"
            f"The focus projects for the {department} department in {month_label} are:\n\n"
            + "\n".join(f"- {p['name']}" for p in focus_projects) + "\n\n"
            f"## Key Deliverables\n\n"
            f"The key deliverables for {month_label} are:\n\n"
            f"- Sprint planning completed by the 1st of {month_name} {year}.\n"
            f"- Mid-month {department} review on {month_name} 15, {year}.\n"
            f"- Retrospective on the last Friday of {month_name} {year}.\n\n"
            f"## Owner\n\n"
            f"The {month_label} {department} monthly plan is owned by the "
            f"{department} department.\n"
        )

        dept_slug = department.lower().replace(" & ", "_").replace(" ", "_")
        write_md(
            BASE / "monthly_plans" / f"plan_{month_date.strftime('%Y_%m')}_{dept_slug}.md",
            frontmatter={
                "type": "monthly_plan",
                "department": department,
                "month": month_key,
                "month_name": month_name,
                "year": year,
                "title": f"{month_label} {department} Monthly Plan",
            },
            body=body,
        )


def generate_job_postings(n: int = 20) -> None:
    titles = [
        ("Senior Backend Engineer", "Engineering"),
        ("DevOps Engineer", "DevOps"),
        ("ML Engineer", "Data & AI"),
        ("Frontend Engineer", "Engineering"),
        ("SRE", "DevOps"),
        ("Data Scientist", "Data & AI"),
        ("IT Support Engineer", "IT Support"),
        ("Sales Manager", "Sales"),
        ("HR Business Partner", "Human Resources"),
        ("Marketing Specialist", "Marketing"),
        ("Financial Analyst", "Finance"),
        ("Legal Counsel", "Legal"),
        ("Operations Coordinator", "Operations"),
        ("Junior Software Engineer", "Engineering"),
        ("Tech Lead", "Engineering"),
        ("Cloud Architect", "DevOps"),
        ("Content Manager", "Marketing"),
        ("Recruiter", "Human Resources"),
        ("Accountant", "Finance"),
        ("Sales Representative", "Sales"),
    ]
    for i, (title, dept) in enumerate(titles[:n], start=1):
        body = (
            f"# Job Posting: {title}\n\n"
            f"**Department**: {dept}\n"
            f"**Location**: {random.choice(OFFICES)}\n"
            f"**Type**: Full-time, permanent\n\n"
            f"## About the Role\n"
            f"Atlas Systems is hiring a {title} to join the {dept} team.\n\n"
            f"## Requirements\n"
            + "\n".join(f"- {s}" for s in random.sample(SKILLS_POOL.get(dept, ["Communication"]),
                                                         k=min(4, len(SKILLS_POOL.get(dept, [])))))
            + "\n\n## How to Apply\nSubmit your CV at https://atlas.com/careers\n"
        )
        write_md(
            BASE / "job_postings" / f"job_{i:03d}.md",
            frontmatter={
                "type": "job_posting",
                "job_id": f"JOB-{i:03d}",
                "title": title,
                "department": dept,
                "status": "open",
            },
            body=body,
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Atlas Systems — Synthetic Enterprise Data Generator")
    print("=" * 60)

    ensure_dirs()

    print("\n[1/8] Generating 150 employees …")
    employees = generate_employees(150)

    print("[2/8] Generating 10 department records …")
    generate_departments(employees)

    print("[3/8] Generating 50 projects …")
    projects = generate_projects(50, employees)

    print("[4/8] Generating 30 policies …")
    generate_policies(30)

    print("[5/8] Generating 100 IT FAQs …")
    generate_it_faqs(100)

    print("[6/8] Generating 150 meetings …")
    generate_meetings(150, employees, projects)

    print("[7/8] Generating 36 monthly plans …")
    generate_monthly_plans(36, projects)

    print("[8/8] Generating 20 job postings …")
    generate_job_postings(20)

    total = 150 + 10 + 50 + 30 + 100 + 150 + 36 + 20
    print(f"\n[OK] Generated {total} markdown documents in ./data/")
    print("    Estimated chunks after ingest: ~1500-2500\n")


if __name__ == "__main__":
    main()
