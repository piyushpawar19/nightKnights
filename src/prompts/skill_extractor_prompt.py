from textwrap import dedent

class SkillExtractorPrompt:
    def __init__(self):
        self.template = dedent("""
        You are an expert AI assistant specialized in extracting and categorizing skills from job descriptions.
        Your task is to analyze the provided job description text and identify all relevant skills.
        Categorize each extracted skill into one of the following predefined categories:

        - Programming Languages
        - Frameworks
        - Libraries
        - Databases
        - Cloud Platforms
        - DevOps Tools
        - AI/ML Technologies
        - Data Engineering Tools
        - BI & Analytics Tools
        - Operating Systems
        - Version Control
        - Methodologies
        - Certifications
        - Soft Skills
        - General Technical Skills
        - Other Skills (for anything that doesn\'t fit the above categories)

        For each skill, ensure it is normalized (consistent casing, no duplicates, common abbreviations expanded).
        The output should be a JSON object conforming to the ExtractedSkills schema, with lists of skills for each category.

        Job Description sections for analysis:
        {parsed_jd_sections}

        Example of desired JSON output format (ExtractedSkills schema):
        ```json
        {
            "programming_languages": ["Python", "Java"],
            "frameworks": ["React", "Django"],
            "libraries": ["TensorFlow"],
            "databases": ["PostgreSQL"],
            "cloud_platforms": ["AWS"],
            "devops_tools": ["Docker", "Kubernetes"],
            "ai_ml": ["Machine Learning", "NLP"],
            "data_engineering": ["Apache Spark"],
            "analytics_bi": ["Tableau"],
            "operating_systems": ["Linux"],
            "version_control": ["Git"],
            "methodologies": ["Agile", "Scrum"],
            "certifications": ["PMP"],
            "soft_skills": ["Communication", "Teamwork"],
            "technical_skills": ["Microservices", "API Design"],
            "other_skills": []
        }
        ```

        Strictly follow the output schema. Do not include any conversational text in your response.
        """)

    def get_prompt(self, parsed_jd_sections: str) -> str:
        return self.template.format(parsed_jd_sections=parsed_jd_sections)
