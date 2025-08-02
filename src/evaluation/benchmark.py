# src/evaluation/benchmark.py
import logging
from typing import List, Dict, Any, Optional
import json
import random
from pathlib import Path
import pandas as pd
import re

logger = logging.getLogger(__name__)

class BenchmarkGenerator:
    def __init__(self):
        """Inicializa gerador de benchmark para construção civil"""
        logger.info("Inicializando gerador de benchmark")
        
        # Templates de perguntas por categoria
        self.question_templates = {
            'regulamentacao': [
                "Quais são as normas de {} em obras urbanas?",
                "Como implementar {} conforme a legislação?",
                "Que documentos são necessários para {}?",
                "Quais são os limites legais para {}?",
                "Como obter licença para {}?"
            ],
            'seguranca': [
                "Que equipamentos de proteção são obrigatórios para {}?",
                "Como treinar equipes para {}?",
                "Quais são os riscos associados a {}?",
                "Como prevenir acidentes em {}?",
                "Que procedimentos seguir durante {}?"
            ],
            'meio_ambiente': [
                "Como minimizar o impacto ambiental de {}?",
                "Que medidas sustentáveis aplicar em {}?",
                "Como gerenciar resíduos de {}?",
                "Qual o impacto de {} nos recursos naturais?",
                "Como implementar {} de forma ecológica?"
            ],
            'qualidade': [
                "Como garantir a qualidade em {}?",
                "Que testes são necessários para {}?",
                "Como monitorar {} durante a execução?",
                "Quais são os padrões de qualidade para {}?",
                "Como documentar {} adequadamente?"
            ],
            'custos': [
                "Como calcular os custos de {}?",
                "Que fatores influenciam o preço de {}?",
                "Como otimizar custos em {}?",
                "Qual o orçamento típico para {}?",
                "Como controlar gastos com {}?"
            ]
        }
        
        # Palavras-chave por categoria
        self.domain_keywords = {
            'regulamentacao': ['monitoramento de ruído', 'licenciamento ambiental', 'alvará de construção', 
                             'normas ABNT', 'código de obras', 'aprovação de projetos'],
            'seguranca': ['EPIs', 'treinamento de segurança', 'trabalho em altura', 'sinalização de obras',
                         'primeiros socorros', 'prevenção de acidentes'],
            'meio_ambiente': ['gestão de resíduos', 'controle de erosão', 'proteção de mananciais',
                            'licenciamento ambiental', 'sustentabilidade', 'impacto ambiental'],
            'qualidade': ['controle de qualidade', 'inspeção de materiais', 'testes de resistência',
                         'certificação', 'normas técnicas', 'documentação'],
            'custos': ['orçamento de obra', 'controle de custos', 'cronograma físico-financeiro',
                      'análise de viabilidade', 'composição de preços', 'margem de lucro']
        }
    
    def generate_qa_from_document(self, document: str, num_questions: int = 3) -> List[Dict]:
        """
        Gera perguntas e respostas baseadas em um documento
        
        Args:
            document: Texto do documento
            num_questions: Número de perguntas a gerar
            
        Returns:
            Lista de dicionários com perguntas e respostas
        """
        qa_pairs = []
        
        if not document or len(document.strip()) < 50:
            logger.warning("Documento muito curto para gerar Q&A")
            return qa_pairs
        
        # Extrai conceitos principais do documento
        key_concepts = self._extract_key_concepts(document)
        
        # Identifica categoria do documento
        category = self._identify_category(document)
        
        for i in range(num_questions):
            try:
                # Gera pergunta baseada no documento
                question = self._generate_question_from_text(document, key_concepts, category)
                
                # Gera resposta baseada no documento
                answer = self._generate_answer_from_text(document, question, key_concepts)
                
                if question and answer:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': 'synthetic',
                        'category': category,
                        'concepts': key_concepts[:3],  # Top 3 conceitos
                        'difficulty': 'medium'
                    })
                    
            except Exception as e:
                logger.warning(f"Erro ao gerar Q&A {i+1}: {e}")
                continue
        
        logger.info(f"Gerados {len(qa_pairs)} pares Q&A do documento")
        return qa_pairs
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extrai conceitos-chave do texto"""
        concepts = []
        
        # Busca por palavras-chave do domínio
        text_lower = text.lower()
        
        for category, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    concepts.append(keyword)
        
        # Extrai substantivos importantes (heurística simples)
        # Palavras com 4+ caracteres que aparecem no início de frases
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            words = sentence.strip().split()
            if words:
                first_words = words[:3]  # Primeiras palavras da frase
                for word in first_words:
                    clean_word = re.sub(r'[^\w\s]', '', word).strip()
                    if len(clean_word) >= 4 and clean_word.lower() not in ['como', 'para', 'deve', 'será']:
                        concepts.append(clean_word)
        
        # Remove duplicatas e retorna top conceitos
        unique_concepts = list(dict.fromkeys(concepts))  # Preserva ordem
        return unique_concepts[:10]
    
    def _identify_category(self, text: str) -> str:
        """Identifica categoria do documento"""
        text_lower = text.lower()
        
        category_scores = {}
        
        for category, keywords in self.domain_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            category_scores[category] = score
        
        # Retorna categoria com maior score, ou 'geral' se empate
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            return best_category if category_scores[best_category] > 0 else 'geral'
        
        return 'geral'
    
    def _generate_question_from_text(self, text: str, concepts: List[str], category: str) -> str:
        """Gera pergunta baseada no texto e conceitos"""
        
        # Seleciona template baseado na categoria
        templates = self.question_templates.get(category, [
            "Como implementar {}?",
            "Quais são os requisitos para {}?",
            "Como funciona o processo de {}?"
        ])
        
        if not concepts:
            # Fallback se não há conceitos
            return "Como implementar as práticas descritas no documento?"
        
        # Seleciona conceito principal e template
        main_concept = concepts[0]
        template = random.choice(templates)
        
        try:
            question = template.format(main_concept)
            return question
        except:
            return f"Como implementar {main_concept} adequadamente?"
    
    def _generate_answer_from_text(self, text: str, question: str, concepts: List[str]) -> str:
        """Gera resposta baseada no texto do documento"""
        
        # Busca frases relevantes no texto
        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = []
        
        # Busca frases que contêm conceitos da pergunta
        question_words = set(question.lower().split())
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Ignora frases muito curtas
                continue
                
            sentence_words = set(sentence.lower().split())
            
            # Calcula relevância baseada em palavras em comum
            common_words = question_words.intersection(sentence_words)
            concept_matches = sum(1 for concept in concepts[:3] if concept.lower() in sentence.lower())
            
            relevance_score = len(common_words) + concept_matches * 2
            
            if relevance_score >= 2:  # Limiar de relevância
                relevant_sentences.append((sentence, relevance_score))
        
        # Ordena por relevância e pega as melhores
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in relevant_sentences[:3]]
        
        if top_sentences:
            # Combina frases relevantes em uma resposta
            answer = '. '.join(top_sentences)
            # Limita tamanho da resposta
            if len(answer) > 500:
                answer = answer[:497] + "..."
            return answer
        else:
            # Fallback: usa início do documento
            return text[:200] + "..." if len(text) > 200 else text
    
    def generate_adversarial_questions(self, domain_keywords: List[str], 
                                     num_questions: int = 5) -> List[Dict]:
        """
        Gera perguntas adversariais/desafiadoras
        
        Args:
            domain_keywords: Palavras-chave do domínio
            num_questions: Número de perguntas a gerar
            
        Returns:
            Lista de perguntas adversariais
        """
        
        adversarial_templates = [
            "Como resolver problemas de {} quando não há dados históricos?",
            "Quais são as limitações de {} em projetos complexos?",
            "Compare {} com abordagens alternativas em termos de custo-benefício",
            "Que problemas podem surgir ao implementar {} em larga escala?",
            "Como {} se comporta sob condições adversas ou atípicas?",
            "Quais são os riscos ocultos de {} que não são óbvios?",
            "Como adaptar {} para contextos diferentes do padrão?",
            "Que aspectos de {} são frequentemente negligenciados?",
            "Como {} interage com outros sistemas na prática?",
            "Quais são as implicações de longo prazo de {}?"
        ]
        
        questions = []
        
        # Gera perguntas usando keywords disponíveis
        available_keywords = domain_keywords if domain_keywords else [
            'monitoramento de ruído', 'controle de qualidade', 'gestão de resíduos',
            'segurança do trabalho', 'licenciamento ambiental'
        ]
        
        for i in range(num_questions):
            template = random.choice(adversarial_templates)
            keyword = random.choice(available_keywords)
            
            question = template.format(keyword)
            
            questions.append({
                'question': question,
                'answer': None,  # Para ser preenchido manualmente
                'type': 'adversarial',
                'difficulty': 'hard',
                'category': 'desafio',
                'keywords': [keyword],
                'requires_expertise': True
            })
        
        logger.info(f"Geradas {len(questions)} perguntas adversariais")
        return questions
    
    def generate_seed_questions(self, categories: List[str] = None) -> List[Dict]:
        """
        Gera perguntas seed baseadas nas categorias
        
        Args:
            categories: Lista de categorias a incluir
            
        Returns:
            Lista de perguntas seed
        """
        if categories is None:
            categories = list(self.question_templates.keys())
        
        seed_questions = []
        
        for category in categories:
            keywords = self.domain_keywords.get(category, [])
            templates = self.question_templates.get(category, [])
            
            if not keywords or not templates:
                continue
            
            # Gera 2 perguntas por categoria
            for i in range(2):
                keyword = random.choice(keywords)
                template = random.choice(templates)
                
                question = template.format(keyword)
                
                seed_questions.append({
                    'question': question,
                    'answer': None,  # Para ser preenchido com base nos documentos
                    'type': 'seed',
                    'category': category,
                    'difficulty': 'medium',
                    'keywords': [keyword]
                })
        
        logger.info(f"Geradas {len(seed_questions)} perguntas seed")
        return seed_questions
    
    def create_golden_dataset(self, documents: List[str], 
                            seed_questions: List[str] = None,
                            output_path: Path = None) -> Dict:
        """
        Cria dataset golden combinando geração automática e curadoria
        
        Args:
            documents: Lista de documentos para gerar Q&A
            seed_questions: Lista de perguntas seed
            output_path: Caminho para salvar o dataset
            
        Returns:
            Dataset golden completo
        """
        logger.info("Criando dataset golden...")
        
        golden_dataset = {
            'metadata': {
                'total_documents': len(documents),
                'generation_date': pd.Timestamp.now().isoformat(),
                'version': '1.0',
                'generator': 'BenchmarkGeneratorV1',
                'domain': 'construcao_civil'
            },
            'synthetic_qa': [],
            'seed_questions': [],
            'adversarial_questions': [],
            'categories': list(self.question_templates.keys())
        }
        
        # 1. Gera Q&A sintético dos documentos
        logger.info("Gerando Q&A sintético dos documentos...")
        for i, doc in enumerate(documents[:15]):  # Limita para evitar excesso
            if not doc or len(str(doc).strip()) < 50:
                continue
                
            qa_pairs = self.generate_qa_from_document(str(doc), num_questions=2)
            for qa in qa_pairs:
                qa['document_id'] = i
                qa['source_document_preview'] = str(doc)[:100] + "..."
                golden_dataset['synthetic_qa'].append(qa)
        
        # 2. Gera perguntas seed estruturadas
        logger.info("Gerando perguntas seed...")
        seed_qa = self.generate_seed_questions()
        golden_dataset['seed_questions'] = seed_qa
        
        # 3. Adiciona perguntas seed customizadas se fornecidas
        if seed_questions:
            logger.info(f"Adicionando {len(seed_questions)} perguntas seed customizadas...")
            for question in seed_questions:
                golden_dataset['seed_questions'].append({
                    'question': question,
                    'answer': None,
                    'type': 'custom_seed',
                    'difficulty': 'medium'
                })
        
        # 4. Gera perguntas adversariais
        logger.info("Gerando perguntas adversariais...")
        all_keywords = []
        for keywords in self.domain_keywords.values():
            all_keywords.extend(keywords)
        
        adversarial = self.generate_adversarial_questions(all_keywords, num_questions=8)
        golden_dataset['adversarial_questions'] = adversarial
        
        # 5. Adiciona exemplos de respostas esperadas
        golden_dataset['answer_examples'] = [
            {
                'question': "Como monitorar ruído em obras urbanas?",
                'expected_answer': "O monitoramento deve seguir NBR 10151, com medidores calibrados e registros horários. Limites: 70dB dia, 60dB noite em áreas residenciais.",
                'keywords': ['NBR 10151', 'medidores', '70dB', '60dB'],
                'concepts': ['monitoramento', 'ruído', 'normas'],
                'forbidden_words': ['opcional', 'não necessário']
            },
            {
                'question': "Que EPIs são obrigatórios em canteiros?",
                'expected_answer': "Capacete, óculos de proteção, luvas, calçados de segurança, cintos para altura. Empresa deve fornecer gratuitamente e treinar trabalhadores.",
                'keywords': ['capacete', 'óculos', 'luvas', 'calçados', 'cintos'],
                'concepts': ['EPIs', 'segurança', 'obrigatório'],
                'forbidden_words': ['opcional', 'responsabilidade do trabalhador']
            }
        ]
        
        # 6. Estatísticas do dataset
        golden_dataset['statistics'] = {
            'total_synthetic_qa': len(golden_dataset['synthetic_qa']),
            'total_seed_questions': len(golden_dataset['seed_questions']),
            'total_adversarial': len(golden_dataset['adversarial_questions']),
            'total_questions': (len(golden_dataset['synthetic_qa']) + 
                              len(golden_dataset['seed_questions']) + 
                              len(golden_dataset['adversarial_questions'])),
            'categories_covered': len(set(qa.get('category', 'unknown') 
                                        for qa in golden_dataset['synthetic_qa']))
        }
        
        # 7. Salva se caminho fornecido
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(golden_dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Dataset golden salvo em {output_path}")
        
        logger.info(f"✅ Dataset golden criado com {golden_dataset['statistics']['total_questions']} perguntas")
        return golden_dataset
    
    def validate_golden_dataset(self, dataset: Dict) -> Dict[str, Any]:
        """
        Valida qualidade do dataset golden
        
        Args:
            dataset: Dataset golden
            
        Returns:
            Relatório de validação
        """
        validation_report = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        try:
            # Verifica estrutura básica
            required_keys = ['synthetic_qa', 'seed_questions', 'adversarial_questions', 'metadata']
            for key in required_keys:
                if key not in dataset:
                    validation_report['errors'].append(f"Chave obrigatória ausente: {key}")
                    validation_report['valid'] = False
            
            # Verifica qualidade das perguntas
            all_questions = []
            
            for qa_list_name in ['synthetic_qa', 'seed_questions', 'adversarial_questions']:
                qa_list = dataset.get(qa_list_name, [])
                
                for i, qa in enumerate(qa_list):
                    if not qa.get('question'):
                        validation_report['warnings'].append(f"Pergunta vazia em {qa_list_name}[{i}]")
                    else:
                        all_questions.append(qa['question'])
            
            # Verifica duplicatas
            unique_questions = set(all_questions)
            if len(unique_questions) < len(all_questions):
                duplicates = len(all_questions) - len(unique_questions)
                validation_report['warnings'].append(f"{duplicates} perguntas duplicadas encontradas")
            
            # Estatísticas
            validation_report['statistics'] = {
                'total_questions': len(all_questions),
                'unique_questions': len(unique_questions),
                'avg_question_length': np.mean([len(q.split()) for q in all_questions]) if all_questions else 0,
                'categories_represented': len(set(qa.get('category', 'unknown') 
                                                for qa in dataset.get('synthetic_qa', [])))
            }
            
            logger.info(f"✅ Validação concluída: {len(validation_report['errors'])} erros, {len(validation_report['warnings'])} warnings")
            
        except Exception as e:
            validation_report['valid'] = False
            validation_report['errors'].append(f"Erro na validação: {str(e)}")
        
        return validation_report
    