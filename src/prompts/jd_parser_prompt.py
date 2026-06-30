JD_PARSER_PROMPT_TEMPLATE = """
You are an advanced AI assistant specializing in parsing Job Descriptions (JDs).
Your task is to extract structured information from the provided raw job description text.

Here is the raw job description:

---
{raw_jd}
---

Extract the following information and present it in a JSON format that conforms to the ParsedJD Pydantic schema.
Ensure all fields are correctly identified and populated based on the schema definitions.

Schema to follow (Pydantic):

class JobInfo(BaseModel):
    title: Optional[str]
    company: Optional[str]
    location: Optional[str]
    employment_type: Optional[str]
    remote_type: Optional[str]
    seniority: Optional[str]
    industry: Optional[str]
    domain: Optional[str]
    salary: Optional[Dict[str, Any]] # {{'min_salary': float, 'max_salary': float, 'currency': str, 'period': str, 'raw_text': str}}
    minimum_experience: Optional[int]
    maximum_experience: Optional[int]

class Education(BaseModel):
    degree: Optional[str]
    field: Optional[str]
    required: bool

class Certification(BaseModel):
    name: str
    required: bool

class Requirements(BaseModel):
    mandatory_requirements: List[str]
    preferred_requirements: List[str]
    certifications: List[Certification]
    education: List[Education]

class Skills(BaseModel):
    technical_skills: List[str]
    programming_languages: List[str]
    frameworks: List[str]
    libraries: List[str]
    databases: List[str]
    cloud: List[str]
    devops: List[str]
    ai_ml: List[str]
    soft_skills: List[str]

class Responsibilities(BaseModel):
    responsibilities_list: List[str]

class Preferences(BaseModel):
    pass

class ParsingMetadata(BaseModel):
    parse_timestamp: str
    parser_version: str

class ParsedJD(BaseModel):
    job_info: JobInfo
    requirements: Requirements
    skills: Skills
    responsibilities: Responsibilities
    preferences: Preferences
    metadata: ParsingMetadata


Consider the following guidelines for extraction:

1.  **Section Detection**: Identify logical sections like Responsibilities, Requirements, Qualifications, Skills, Education, Benefits, etc. even if headings are flexible.
2.  **Experience**: Extract minimum and maximum years of experience (e.g., "5+ years", "2-5 years", "minimum of 3 years"). Map seniority terms (junior, mid, senior, lead, staff, principal, architect) if present.
3.  **Education**: Recognize degrees (BE, BTech, BS, ME, MTech, MS, MBA, PhD) and equivalent qualifications. Identify the field of study if specified.
4.  **Salary**: Extract salary ranges (e.g., "$100k - $120k per annum", "8-12 LPA"), currency (USD, INR, etc.), and period (annual/monthly).
5.  **Requirements**: Differentiate between mandatory ("must have", "required") and preferred ("nice to have", "bonus points") requirements. Extract certifications.
6.  **Skills**: Categorize technical skills including programming languages, frameworks, libraries, databases, cloud platforms, DevOps tools, AI/ML tools, and soft skills.
7.  **Responsibilities**: List the key duties and responsibilities.

Present the output strictly as a JSON object, without any additional explanatory text or markdown outside the JSON.
"""
