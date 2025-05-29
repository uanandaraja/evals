import json
import os
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
import re

def load_jsonl(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def evaluate_mcq_reasoning(item, model, client):
    """Evaluate a single multiple choice question with reasoning model"""
    prompt = f"""Ini adalah soal {item['subject']} untuk {item['level']}. Pilihlah salah satu jawaban yang dianggap benar!

{item['soal']}
{item['jawaban']}

Jawab HANYA dengan huruf pilihan saja (A, B, C, D, atau E). Jangan tambahkan penjelasan awal atau akhir. Hanya output huruf pilihan saja.
Jawaban:"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    
    # Extract both reasoning content and final answer
    reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
    predicted = response.choices[0].message.content.strip()
    correct = item['kunci']
    
    # Extract only the final letter from the predicted answer
    # Look for single letters A, B, C, D, E in the final answer
    letter_match = re.search(r'\b([ABCDE])\b', predicted)
    if letter_match:
        predicted_letter = letter_match.group(1)
    else:
        # Fallback: take the first character if it's a valid option
        predicted_letter = predicted[0] if predicted and predicted[0] in 'ABCDE' else predicted
    
    return {
        'id': item['id'],
        'predicted': predicted_letter,
        'correct': correct,
        'is_correct': predicted_letter == correct,
        'reasoning_content': reasoning_content,
        'reasoning_length': len(reasoning_content) if reasoning_content else 0,
        'full_response': predicted  # Keep the full response for debugging
    }

def main():
    # Configure OpenAI client for OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set!")
        print("Please set it with: export OPENROUTER_API_KEY=your_api_key")
        return
    
    # Reasoning models to evaluate
    reasoning_models = [
        "deepseek/deepseek-r1-0528"
    ]
    
    # Load and filter data
    print("Loading data...")
    try:
        data = load_jsonl('indoMMLU.jsonl')
    except FileNotFoundError:
        print("Error: indoMMLU.jsonl file not found!")
        print("Please make sure the file exists in the current directory.")
        return
    
    filtered_data = [item for item in data if item['level'] == 'Seleksi PTN' and item['is_for_fewshot'] == '0']
    print(f"Loaded {len(filtered_data)} questions for evaluation")
    
    all_model_results = {}
    
    for model in reasoning_models:
        print(f"\n{'='*50}")
        print(f"Evaluating reasoning model: {model}")
        print(f"{'='*50}")
        
        results = []
        correct_count = 0
        total_reasoning_tokens = 0
        
        for i, item in enumerate(tqdm(filtered_data, desc=f"Evaluating {model}")):
            try:
                result = evaluate_mcq_reasoning(item, model, client)
                
                # Add more details to result
                result.update({
                    'model': model,
                    'subject': item['subject'],
                    'soal': item['soal'],
                    'jawaban': item['jawaban'],
                    'sumber': item['sumber']
                })
                
                results.append(result)
                
                if result['is_correct']:
                    correct_count += 1
                
                total_reasoning_tokens += result['reasoning_length']
                current_accuracy = correct_count / len(results)
                avg_reasoning_length = total_reasoning_tokens / len(results)
                
                # Show first 10 outputs with reasoning info
                if i < 10:
                    status = "✓" if result['is_correct'] else "✗"
                    print(f"\nQuestion {i+1} ({item['subject']}):")
                    print(f"Predicted: {result['predicted']} | Correct: {result['correct']} {status}")
                    print(f"Full response: {result['full_response'][:100]}...")
                    print(f"Question: {item['soal'][:100]}...")
                    print(f"Options: {item['jawaban']}")
                    print(f"Reasoning length: {result['reasoning_length']} chars")
                    if result['reasoning_content'] and len(result['reasoning_content']) > 0:
                        print(f"Reasoning preview: {result['reasoning_content'][:200]}...")
                    print(f"Running accuracy: {current_accuracy:.3f}")
                    print(f"Avg reasoning length: {avg_reasoning_length:.1f}")
                    print("-" * 60)
                
                # Show progress every 50 questions
                if (i + 1) % 50 == 0:
                    print(f"\nProgress: {i+1}/{len(filtered_data)} | Accuracy: {current_accuracy:.3f} | Avg reasoning: {avg_reasoning_length:.1f}")
                    
            except Exception as e:
                print(f"Error evaluating question {i+1}: {e}")
                continue
        
        # Save results for this model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model.replace('/', '_').replace(':', '_')
        output_file = f"eval_results_reasoning_{model_name}_{timestamp}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Calculate statistics
        final_accuracy = sum(r['is_correct'] for r in results) / len(results) if results else 0
        avg_reasoning_length = sum(r['reasoning_length'] for r in results) / len(results) if results else 0
        reasoning_usage_rate = sum(1 for r in results if r['reasoning_length'] > 0) / len(results) if results else 0
        
        all_model_results[model] = {
            'accuracy': final_accuracy,
            'avg_reasoning_length': avg_reasoning_length,
            'reasoning_usage_rate': reasoning_usage_rate,
            'results': results,
            'output_file': output_file
        }
        
        print(f"Final accuracy for {model}: {final_accuracy:.3f}")
        print(f"Average reasoning length: {avg_reasoning_length:.1f} characters")
        print(f"Reasoning usage rate: {reasoning_usage_rate:.1%}")
        print(f"Results saved to: {output_file}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON - REASONING MODELS")
    print(f"{'='*60}")
    for model, data in all_model_results.items():
        print(f"{model}:")
        print(f"  Accuracy: {data['accuracy']:.3f}")
        print(f"  Avg reasoning length: {data['avg_reasoning_length']:.1f}")
        print(f"  Reasoning usage: {data['reasoning_usage_rate']:.1%}")
        print()

if __name__ == "__main__":
    main() 