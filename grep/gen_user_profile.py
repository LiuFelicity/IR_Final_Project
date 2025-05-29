import random
import json

# 定義年齡層與地區
ages = ['Freshman/Sophomore', 'Junior/Senior', 'Master', 'PhD']
countries = ['Europe', 'America', 'Asia']

# 定義所有興趣
interests = [
    'Cross-cultural exchange', 'Public affairs', 'Art', 'Photography', 'Artificial Intelligence',
    'Media', 'Language learning', 'Skill learning', 'Online learning', 'Education',
    'Environmental sustainability', 'Business', 'Entrepreneurship', 'Psychology',
    'Programming skills', 'Communication and writing', 'Science'
]

# 定義部門與對應興趣
department_interest_map = {
    'Department of Chinese Literature': 'Communication and writing',
    'Department of Foreign Languages and Literatures': 'Language learning',
    'Department of History': 'Public affairs',
    'Department of Philosophy': 'Public affairs',
    'Department of Anthropology': 'Cross-cultural exchange',
    'Department of Library and Information Science': 'Online learning',
    'Department of Japanese Language and Literature': 'Language learning',
    'Department of Drama and Theatre': 'Art',
    'Department of Mathematics': 'Science',
    'Department of Physics': 'Science',
    'Department of Chemistry': 'Science',
    'Department of Geosciences': 'Science',
    'Department of Psychology': 'Psychology',
    'Department of Geography': 'Environmental sustainability',
    'Department of Atmospheric Sciences': 'Science',
    'Department of Political Science': 'Public affairs',
    'Department of Economics': 'Business',
    'Department of Sociology': 'Cross-cultural exchange',
    'Department of Social Work': 'Psychology',
    'Department of Medicine': 'Education',
    'Department of Dentistry': 'Education',
    'Department of Pharmacy': 'Education',
    'Department of Clinical Laboratory Sciences and Medical Biotechnology': 'Science',
    'Department of Nursing': 'Psychology',
    'Department of Physical Therapy': 'Skill learning',
    'Department of Occupational Therapy': 'Psychology',
    'Department of Civil Engineering': 'Skill learning',
    'Department of Mechanical Engineering': 'Skill learning',
    'Department of Chemical Engineering': 'Skill learning',
    'Department of Engineering Science and Ocean Engineering': 'Skill learning',
    'Department of Materials Science and Engineering': 'Skill learning',
    'Department of Agronomy': 'Environmental sustainability',
    'Department of Bioenvironmental Systems Engineering': 'Environmental sustainability',
    'Department of Agricultural Chemistry': 'Environmental sustainability',
    'Department of Forestry and Resource Conservation': 'Environmental sustainability',
    'Department of Animal Science and Technology': 'Environmental sustainability',
    'Department of Agricultural Economics': 'Business',
    'Department of Horticultural Science': 'Environmental sustainability',
    'Department of Veterinary Medicine': 'Environmental sustainability',
    'Department of Bio-Industry Communication and Development': 'Media',
    'Department of Bio-industrial Mechatronics Engineering': 'Skill learning',
    'Department of Entomology': 'Environmental sustainability',
    'Department of Plant Pathology and Microbiology': 'Environmental sustainability',
    'Department of Business Administration': 'Business',
    'Department of Accounting': 'Business',
    'Department of Finance': 'Business',
    'Department of International Business': 'Cross-cultural exchange',
    'Department of Information Management': 'Artificial Intelligence',
    'Department of Public Health': 'Education',
    'Department of Electrical Engineering': 'Artificial Intelligence',
    'Department of Computer Science and Information Engineering': 'Programming skills',
    'Department of Law': 'Public affairs',
    'Department of Life Science': 'Science',
    'Department of Biochemical Science and Technology': 'Science'
}

departments = list(department_interest_map.keys())

# 生成用戶資料
user_profiles = []
for i in range(5000):
    dept = random.choice(departments)
    main_interest = department_interest_map[dept]

    other_interests = list(set(interests) - {main_interest})
    sampled = random.sample(other_interests, k=random.randint(0, 2))
    user_interests = [main_interest] + sampled
    random.shuffle(user_interests)

    age = random.choice(ages)
    user_countries = random.sample(countries, random.randint(1, 3))

    user_profiles.append({
        'id': f'user_{i+1}',
        'department': dept,
        'age': age,
        'interests': user_interests,
        'countries': user_countries
    })

# 儲存成 JSON
with open('pseudo_user_profiles.json', 'w', encoding='utf-8') as f:
    json.dump(user_profiles, f, ensure_ascii=False, indent=2)
