import numpy as np
import os # Added for path joining, though not strictly necessary if paths are hardcoded relative to CWD

# Paths assuming the script is run from the project root (e.g., IR_Final_Project/)
# and .npz files are in the 'grep' subdirectory.
department_path = "department.txt"
term_data_path = "term_data_lsi.npz"  # Updated from terms_lsi.npy
doc_data_path = "doc_data_lsi.npz"    # Updated from doc_vectors_lsi.npy
output_path = "departments_lsi.npy"

def read_departments():
    """
    Reads the department names from a file and returns them as a list of word lists.
    Each inner list contains the lowercase words of a department name.
    """
    departments_word_lists = []
    try:
        with open(department_path, 'r', encoding='utf-8') as file: # Added encoding
            for line in file:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                words = line.split()
                # Assuming department name starts from the 3rd word (index 2)
                if len(words) > 2:
                    departments_word_lists.append([word.lower() for word in words[2:]])
                # else:
                    # print(f"Skipping line due to insufficient words for department name: {line}")
    except FileNotFoundError:
        print(f"Error: Department file not found at {department_path}")
    return departments_word_lists

def read_term_data():
    """
    Loads terms and their LSI vectors from an .npz file.
    Returns a dictionary mapping terms to vectors.
    """
    try:
        data = np.load(term_data_path, allow_pickle=True)
        terms = data['terms']
        vectors = data['vectors']
        term_to_vector_map = {term: vectors[i] for i, term in enumerate(terms)}
        return term_to_vector_map
    except FileNotFoundError:
        print(f"Error: Term data file not found at {term_data_path}")
        return {}
    except KeyError:
        print(f"Error: 'terms' or 'vectors' key not found in {term_data_path}")
        return {}

def read_opportunity_data():
    """
    Loads opportunity file names and their LSI vectors from an .npz file.
    Returns (file_names array, vectors matrix).
    """
    try:
        data = np.load(doc_data_path, allow_pickle=True)
        file_names = data['file_names']
        vectors = data['vectors']
        return file_names, vectors
    except FileNotFoundError:
        print(f"Error: Opportunity/document data file not found at {doc_data_path}")
        return np.array([]), np.array([])
    except KeyError:
        print(f"Error: 'file_names' or 'vectors' key not found in {doc_data_path}")
        return np.array([]), np.array([])

def write_departments_lsi(departments_word_lists, term_to_vector_map):
    """
    Computes LSI vectors for departments by averaging word vectors and saves them.
    """
    num_departments = len(departments_word_lists)
    if num_departments == 0 or not term_to_vector_map:
        print("No departments or term vectors to process.")
        # Ensure output_path directory exists if we were to save an empty file
        # For now, just return an empty array.
        return np.array([])

    # Infer vector dimension from the first available term vector
    vector_dim = None
    for term_vec in term_to_vector_map.values():
        vector_dim = term_vec.shape[0]
        break
    
    if vector_dim is None:
        print("Could not determine vector dimension from term vectors.")
        return np.array([])

    department_lsi_vectors = np.zeros((num_departments, vector_dim))
    for i, dept_words in enumerate(departments_word_lists):
        accumulated_vector = np.zeros(vector_dim)
        count = 0
        for word in dept_words:
            if word in term_to_vector_map:
                accumulated_vector += term_to_vector_map[word]
                count += 1
        if count > 0:  # Average the vectors
            department_lsi_vectors[i] = accumulated_vector / count
            
    try:
        np.save(output_path, department_lsi_vectors)
        print(f"Department LSI vectors saved for {num_departments} departments to '{output_path}'.")
    except Exception as e:
        print(f"Error saving department LSI vectors to '{output_path}': {e}")
    return department_lsi_vectors

def relevant_opportunities(department_lsi_vectors, opportunity_file_names, opportunity_vectors, departments_word_lists=None):
    """
    Finds and prints top 25 relevant opportunities for each department based on cosine similarity.
    """
    if department_lsi_vectors.size == 0 or opportunity_vectors.size == 0 or opportunity_file_names.size == 0:
        print("Cannot find relevant opportunities due to empty department, opportunity vectors, or file names.")
        return []

    # Normalize department vectors
    norm_department_lsi_vectors = np.array([
        vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec 
        for vec in department_lsi_vectors
    ])

    # Normalize opportunity vectors
    norm_opportunity_vectors = np.array([
        vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
        for vec in opportunity_vectors
    ])

    all_top_opportunities_info = []
    for i in range(len(norm_department_lsi_vectors)):
        dept_vec = norm_department_lsi_vectors[i]
        
        # Skip if department vector is zero (e.g., no words matched, or norm was zero)
        if np.linalg.norm(dept_vec) < 1e-9: # Check against a small epsilon
            print(f"\nDepartment {i} has a zero or near-zero vector, skipping recommendation.")
            all_top_opportunities_info.append([])
            continue

        # Calculate cosine similarities: (N_ops, D_features) dot (D_features,) -> (N_ops,)
        similarities = np.dot(norm_opportunity_vectors, dept_vec)

        # Get top 25 opportunities
        # Argsort sorts in ascending order, so use negative similarities for descending
        num_top_opportunities = min(25, len(opportunity_file_names)) # Ensure we don't ask for more than available
        ranked_indices = np.argsort(-similarities)[:num_top_opportunities]
        
        top_opportunities_for_dept = []
        # Assuming department names could be read from departments_word_lists if needed for print
        # For now, using index i for "Department i"
        dept_name = " ".join(departments_word_lists[i]) if i < len(departments_word_lists) else f"Department {i}"
        print(f"\nTop {num_top_opportunities} opportunities for Department {dept_name}:")
        for rank, opp_idx in enumerate(ranked_indices):
            similarity_score = similarities[opp_idx]
            opp_name = opportunity_file_names[opp_idx]
            print(f"  {rank+1}. {opp_name} (Similarity: {similarity_score:.4f})")
            top_opportunities_for_dept.append({'name': opp_name, 'score': similarity_score, 'index': opp_idx})
        all_top_opportunities_info.append(top_opportunities_for_dept)
        
    return all_top_opportunities_info

def main():
    departments_word_lists = read_departments()
    if not departments_word_lists:
        print("No department data loaded. Exiting.")
        return

    term_to_vector_map = read_term_data()
    if not term_to_vector_map:
        print("No term vector data loaded. Exiting.")
        return

    opportunity_file_names, opportunity_vectors = read_opportunity_data()
    if opportunity_file_names.size == 0 or opportunity_vectors.size == 0:
        print("No opportunity data loaded. Exiting.")
        return

    department_lsi_vectors = write_departments_lsi(departments_word_lists, term_to_vector_map)
    if department_lsi_vectors.size == 0:
        # This could be normal if no departments were processed or no vectors could be generated.
        # The write_departments_lsi function would have printed a message.
        print("Department LSI vectors were not generated or are empty. Cannot proceed with recommendations.")
        return
        
    recommendations = relevant_opportunities(department_lsi_vectors, opportunity_file_names, opportunity_vectors, departments_word_lists)
    # recommendations list can be used for further processing if needed.
    if not recommendations:
        print("No recommendations generated.")

if __name__ == '__main__':
    main()