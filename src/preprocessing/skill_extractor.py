import re
import logging
from typing import List, Dict, Set
from functools import lru_cache

from nightKnights.src.interfaces.skill_interface import SkillExtractionEngineInterface

logger = logging.getLogger(__name__)

class SkillExtractor(SkillExtractionEngineInterface):
    def __init__(self, config: Dict = None):
        self.config = config if config is not None else self._default_config()
        self.skill_patterns: Dict[str, List[re.Pattern]] = self._compile_patterns()
        self.keyword_dictionaries: Dict[str, Set[str]] = self._load_keyword_dictionaries()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def _default_config(self) -> Dict:
        return {
            "sections_to_process": [
                "job_description", "requirements", "responsibilities", "qualifications",
                "skills", "experience", "education", "about_us", "benefits"
            ],
            "skill_keywords": {
                "programming_languages": [
                    "Python", "Java", "JavaScript", "C++", "C#", "Go", "Ruby", "PHP",
                    "TypeScript", "Swift", "Kotlin", "Rust", "Scala", "Perl", "HTML", "CSS",
                    "SQL", "R", "Bash", "Shell"
                ],
                "frameworks": [
                    "React", "Angular", "Vue", "Node.js", "Express", "Django", "Flask",
                    "Spring", "Rails", "ASP.NET", "Laravel", "Symfony", "Next.js",
                    "NestJS", "Nuxt.js", "Gatsby", "Flutter", "Xamarin", "React Native",
                    "jQuery", "Bootstrap", "Tailwind", "Material-UI", "Ant Design"
                ],
                "libraries": [
                    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy",
                    "Matplotlib", "Seaborn", "D3.js", "Redux", "Axios", "Mongoose",
                    "SQLAlchemy", "Hibernate", "NLTK", "SpaCy", "OpenCV", "Pillow",
                    "Requests", "Beautiful Soup", "Selenium", "Cypress", "Jest", "Mocha",
                    "Chai", "JUnit", "NUnit", "XUnit", "Mockito", "pytest"
                ],
                "databases": [
                    "PostgreSQL", "MongoDB", "MySQL", "SQL Server", "Oracle", "DynamoDB",
                    "Cassandra", "Redis", "Elasticsearch", "Firebase", "MariaDB", "SQLite",
                    "Cosmos DB", "Neo4j", "HBase", "Teradata", "Vertica", "Snowflake", "Databricks"
                ],
                "cloud_platforms": [
                    "AWS", "Azure", "GCP", "IBM Cloud", "Oracle Cloud", "Alibaba Cloud",
                    "Heroku", "Vercel", "Netlify", "DigitalOcean"
                ],
                "devops_tools": [
                    "Docker", "Kubernetes", "Jenkins", "Git", "GitHub", "GitLab", "Bitbucket",
                    "Travis CI", "CircleCI", "Ansible", "Puppet", "Chef", "Terraform",
                    "CloudFormation", "Prometheus", "Grafana", "Nagios", "Splunk", "ELK Stack",
                    "Jira", "Confluence", "Trello", "Asana", "Argo CD", "Tekton", "CI/CD"
                ],
                "ai_ml": [
                    "Machine Learning", "Deep Learning", "AI", "NLP", "Computer Vision",
                    "Reinforcement Learning", "Generative AI", "LLM", "Neural Networks",
                    "Data Science", "Predictive Modeling", "Statistical Modeling", "MLOps"
                ],
                "data_engineering": [
                    "Apache Spark", "Apache Hadoop", "Apache Kafka", "ETL", "Data Warehousing",
                    "Data Lakes", "Apache Flink", "Apache Airflow", "Databricks", "Snowflake",
                    "Delta Lake", "Presto", "Trino", "Hive", "Pig", "Kafka Streams", "Beam"
                ],
                "analytics_bi": [
                    "Tableau", "Power BI", "Looker", "QlikView", "Google Analytics", "Mixpanel",
                    "Amplitude", "Metabase", "Redash", "Superset", "Business Intelligence",
                    "Data Visualization", "Reporting", "Dashboards"
                ],
                "operating_systems": [
                    "Linux", "Windows", "macOS", "Ubuntu", "CentOS", "Red Hat", "Debian",
                    "Android", "iOS"
                ],
                "version_control": [
                    "Git", "GitHub", "GitLab", "Bitbucket", "SVN", "Perforce", "Mercurial"
                ],
                "methodologies": [
                    "Agile", "Scrum", "Kanban", "Waterfall", "Lean", "XP", "TDD", "BDD",
                    "DDD", "DevOps"
                ],
                "certifications": [
                    "PMP", "CSM", "AWS Certified", "Azure Certified", "GCP Certified", "CompTIA",
                    "CCNA", "Cisco Certified", "Oracle Certified", "Microsoft Certified",
                    "CKA", "CEH"
                ],
                "soft_skills": [
                    "Communication", "Teamwork", "Problem Solving", "Critical Thinking",
                    "Adaptability", "Leadership", "Time Management", "Collaboration",
                    "Creativity", "Interpersonal Skills", "Attention to Detail"
                ],
                "technical_skills": [
                    "API Design", "Microservices", "System Design", "OOP", "Functional Programming",
                    "Data Structures", "Algorithms", "Networking", "Security", "Scalability",
                    "Performance Tuning", "Database Design", "Web Development", "Mobile Development",
                    "Backend Development", "Frontend Development", "Fullstack Development",
                    "Cloud Architecture", "Containerization", "Virtualization", "Scripting",
                    "Automation", "Troubleshooting", "Debugging", "Technical Writing",
                    "Unit Testing", "Integration Testing", "End-to-End Testing", "CI/CD Pipelines",
                    "Infrastructure as Code", "SRE", "Blockchain", "Cybersecurity"
                ]
            }
        }

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        skill_patterns = {}
        for category, keywords in self.config["skill_keywords"].items():
            patterns = []
            for keyword in keywords:
                pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
                patterns.append(pattern)
            skill_patterns[category] = patterns
        return skill_patterns

    def _load_keyword_dictionaries(self) -> Dict[str, Set[str]]:
        keyword_dicts = {}
        for category, keywords in self.config["skill_keywords"].items():
            keyword_dicts[category] = set(k.lower() for k in keywords)
        return keyword_dicts

    @lru_cache(maxsize=1024) # Cache extracted skills for a given text
    def extract_skills(self, text: str) -> List[str]:
        extracted: Set[str] = set()
        text_lower = text.lower()

        for category, keywords_set in self.keyword_dictionaries.items():
            for keyword in keywords_set:
                if keyword in text_lower:
                    extracted.add(keyword)

        for category, patterns in self.skill_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    extracted.add(match.group(0))

        logging.info(f"Extracted {len(extracted)} raw skills.")
        return list(extracted)

    def extract_from_parsed_jd(self, parsed_jd: Dict) -> List[str]:
        all_extracted_skills: List[str] = []
        for section_name in self.config["sections_to_process"]:
            if section_name in parsed_jd and parsed_jd[section_name]:
                section_content = parsed_jd[section_name]
                if isinstance(section_content, str):
                    all_extracted_skills.extend(self.extract_skills(section_content))
                elif isinstance(section_content, list):
                    for item in section_content:
                        if isinstance(item, str):
                            all_extracted_skills.extend(self.extract_skills(item))
                        elif isinstance(item, dict):
                            for key, value in item.items():
                                if isinstance(value, str):
                                    all_extracted_skills.extend(self.extract_skills(value))
                                elif isinstance(value, list):
                                    for sub_item in value:
                                        if isinstance(sub_item, str):
                                            all_extracted_skills.extend(self.extract_skills(sub_item))

        return all_extracted_skills
