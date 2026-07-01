from typing import Dict, List
from src.interfaces.skill_interface import SkillTaxonomyInterface

class SkillTaxonomy(SkillTaxonomyInterface):
    def __init__(self):
        self.taxonomy: Dict[str, List[str]] = {
            "programming_languages": [
                "Python", "JavaScript", "Java", "C++", "C#", "Go", "Ruby", "PHP",
                "TypeScript", "Swift", "Kotlin", "Rust", "Scala", "Perl", "Shell Scripting",
                "HTML", "CSS", "SQL", "R"
            ],
            "frameworks": [
                "React", "Angular", "Vue.js", "Node.js", "Express.js", "Django", "Flask",
                "Spring", "Ruby on Rails", "ASP.NET", "Laravel", "Symfony", "Next.js",
                "NestJS", "Nuxt.js", "Gatsby", "Flutter", "Xamarin", "React Native",
                "AngularJS", "jQuery", "Bootstrap", "Tailwind CSS", "Material-UI", "Ant Design",
                "Sass", "Less"
            ],
            "libraries": [
                "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy",
                "Matplotlib", "Seaborn", "D3.js", "Lodash", "Moment.js", "RxJS", "Redux",
                "Axios", "Mongoose", "SQLAlchemy", "Hibernate", "NLTK", "SpaCy", "OpenCV",
                "Pillow", "Requests", "Beautiful Soup", "Selenium", "Cypress", "Jest",
                "Mocha", "Chai", "JUnit", "NUnit", "XUnit", "Mockito", "pytest"
            ],
            "databases": [
                "PostgreSQL", "MongoDB", "MySQL", "SQL Server", "Oracle Database", "DynamoDB",
                "Cassandra", "Redis", "Elasticsearch", "Firebase", "MariaDB", "SQLite",
                "Cosmos DB", "Neo4j", "HBase", "Teradata", "Vertica", "Snowflake", "Databricks"
            ],
            "cloud_platforms": [
                "AWS", "Azure", "GCP", "IBM Cloud", "Oracle Cloud", "Alibaba Cloud",
                "Heroku", "Vercel", "Netlify", "DigitalOcean", "Linode", "Rackspace"
            ],
            "devops_tools": [
                "Docker", "Kubernetes", "Jenkins", "Git", "GitHub", "GitLab", "Bitbucket",
                "Travis CI", "CircleCI", "Ansible", "Puppet", "Chef", "Terraform", "CloudFormation",
                "Prometheus", "Grafana", "Nagios", "Splunk", "ELK Stack", "Jira", "Confluence",
                "Trello", "Asana", "PagerDuty", "New Relic", "Datadog", "Argo CD", "Tekton",
                "Azure DevOps", "GitLab CI/CD", "GitHub Actions", "SVN", "Perforce", "TeamCity"
            ],
            "ai_ml": [
                "Machine Learning", "Deep Learning", "Artificial Intelligence",
                "Natural Language Processing", "Computer Vision", "Reinforcement Learning",
                "Generative AI", "Large Language Models", "Neural Networks", "Data Science",
                "Predictive Modeling", "Statistical Modeling", "MLOps", "AutoML", "Prompt Engineering"
            ],
            "data_engineering": [
                "Apache Spark", "Apache Hadoop", "Apache Kafka", "ETL", "Data Warehousing",
                "Data Lakes", "Apache Flink", "Apache Airflow", "Databricks", "Snowflake",
                "Delta Lake", "Presto", "Trino", "Hive", "Pig", "Kafka Streams", "Beam",
                "Azure Data Factory", "AWS Glue", "Google Dataflow", "Informatica", "Talend",
                "DBT", "Fivetran"
            ],
            "analytics_bi": [
                "Tableau", "Power BI", "Looker", "QlikView", "Google Analytics", "Mixpanel",
                "Amplitude", "Metabase", "Redash", "Superset", "SSRS", "SSIS", "SSAS",
                "Business Intelligence", "Data Visualization", "Reporting", "Dashboards",
                "Statistical Analysis", "A/B Testing"
            ],
            "operating_systems": [
                "Linux", "Windows", "macOS", "Ubuntu", "CentOS", "Red Hat", "Debian",
                "Android", "iOS"
            ],
            "version_control": [
                "Git", "GitHub", "GitLab", "Bitbucket", "SVN", "Perforce", "Mercurial"
            ],
            "methodologies": [
                "Agile", "Scrum", "Kanban", "Waterfall", "Lean", "Extreme Programming (XP)",
                "Test-Driven Development (TDD)", "Behavior-Driven Development (BDD)",
                "Domain-Driven Design (DDD)", "DevOps", "SAFe", "Less"
            ],
            "certifications": [
                "PMP", "CSM", "AWS Certified", "Azure Certified", "GCP Certified", "CompTIA",
                "CCNA", "Cisco Certified", "Oracle Certified", "Microsoft Certified",
                "Certified Kubernetes Administrator (CKA)", "Certified Ethical Hacker (CEH)"
            ],
            "soft_skills": [
                "Communication", "Teamwork", "Problem Solving", "Critical Thinking",
                "Adaptability", "Leadership", "Time Management", "Project Management",
                "Collaboration", "Creativity", "Interpersonal Skills", "Attention to Detail",
                "Conflict Resolution", "Negotiation", "Presentation Skills", "Mentorship",
                "Customer Service", "Emotional Intelligence", "Decision Making", "Work Ethic"
            ],
            "technical_skills": [
                "API Design", "Microservices", "System Design", "Object-Oriented Programming (OOP)",
                "Functional Programming", "Data Structures", "Algorithms", "Networking",
                "Security", "Scalability", "Performance Tuning", "Database Design", "Web Development",
                "Mobile Development", "Desktop Development", "Backend Development",
                "Frontend Development", "Fullstack Development", "Cloud Architecture",
                "Containerization", "Virtualization", "Scripting", "Automation", "Troubleshooting",
                "Debugging", "Technical Writing", "Requirement Gathering", "UML", "Design Patterns",
                "User Experience (UX)", "User Interface (UI)", "Responsive Design", "SEO",
                "Unit Testing", "Integration Testing", "End-to-End Testing", "Load Testing",
                "Penetration Testing", "Security Testing", "CI/CD Pipelines", "Infrastructure as Code (IaC)",
                "Site Reliability Engineering (SRE)", "Incident Management", "Disaster Recovery",
                "Blockchain", "Cybersecurity", "Embedded Systems", "Firmware Development",
                "Game Development", "CAD", "CAM", "ERP", "CRM", "SAP", "Salesforce"
            ],
            "other_skills": []
        }

    def categorize_skill(self, skill: str) -> str:
        formatted_skill = self._format_skill_for_taxonomy(skill)

        for category, skills_list in self.taxonomy.items():
            if formatted_skill in skills_list:
                return category

        if "script" in skill.lower() and "shell" not in skill.lower():
            return "programming_languages"
        if "framework" in skill.lower() or "mvc" in skill.lower():
            return "frameworks"
        if "library" in skill.lower() or "sdk" in skill.lower():
            return "libraries"
        if "database" in skill.lower() or "db" in skill.lower() or "sql" in skill.lower() or "nosql" in skill.lower():
            return "databases"
        if "cloud" in skill.lower() or "aws" in skill.lower() or "azure" in skill.lower() or "gcp" in skill.lower():
            return "cloud_platforms"
        if "devops" in skill.lower() or "ci/cd" in skill.lower() or "pipeline" in skill.lower() or "container" in skill.lower() or "orchestration" in skill.lower():
            return "devops_tools"
        if any(ml_term in skill.lower() for ml_term in ["machine learning", "ml", "ai", "deep learning", "nlp", "computer vision"]):
            return "ai_ml"
        if "data engineering" in skill.lower() or "etl" in skill.lower() or "data warehousing" in skill.lower() or "big data" in skill.lower():
            return "data_engineering"
        if "analytics" in skill.lower() or "bi" in skill.lower() or "dashboard" in skill.lower() or "reporting" in skill.lower():
            return "analytics_bi"
        if "os" in skill.lower() or "operating system" in skill.lower() or "linux" in skill.lower() or "windows" in skill.lower() or "macos" in skill.lower():
            return "operating_systems"
        if "version control" in skill.lower() or "git" in skill.lower() or "svn" in skill.lower():
            return "version_control"
        if "agile" in skill.lower() or "scrum" in skill.lower() or "kanban" in skill.lower() or "methodology" in skill.lower():
            return "methodologies"
        if "certified" in skill.lower() or "certification" in skill.lower() or "pmp" in skill.lower() or "csm" in skill.lower():
            return "certifications"
        if any(soft_skill_term in skill.lower() for soft_skill_term in [
            "communication", "teamwork", "problem solving", "critical thinking", "adaptability",
            "leadership", "time management", "collaboration", "creativity", "interpersonal"
        ]):
            return "soft_skills"
        
        technical_keywords = [
            "development", "engineering", "architecture", "design", "security", "testing",
            "backend", "frontend", "fullstack", "programming", "coding", "system", "network"
        ]
        if any(tk in skill.lower() for tk in technical_keywords):
            return "technical_skills"

        return "other_skills"

    def _format_skill_for_taxonomy(self, skill: str) -> str:
        if skill.upper() == "AWS": return "AWS"
        if skill.upper() == "GCP": return "GCP"
        if skill.lower() == "node.js": return "Node.js"
        if skill.upper() == "C#": return "C#"
        if skill.upper() == "C++": return "C++"
        if skill.upper() == "CI/CD": return "CI/CD"
        if skill.upper() == "NLP": return "NLP"
        if skill.upper() == "SQL": return "SQL"
        if skill.upper() == "API": return "API"
        if skill.upper() == "ETL": return "ETL"
        if skill.upper() == "BI": return "BI"
        if skill.upper() == "ML": return "Machine Learning"
        if skill.upper() == "AI": return "Artificial Intelligence"

        words = skill.replace(".", " ").replace("-", " ").split()
        formatted_words = []
        for word in words:
            if word.lower() in ["on", "of", "in", "for", "and", "with", "as", "the", "a", "an", "to"]:
                formatted_words.append(word.lower())
            else:
                formatted_words.append(word.capitalize())
        res = " ".join(formatted_words)
        res = res.replace(" . ", ".").replace(" - ", "-")
        return res
