"""
Sistema Avanzado de Análisis de Sentimientos Basado en Aspectos (ABSA)
Proyecto de Text Mining y Procesamiento del Lenguaje Natural

VERSIÓN FINAL OPTIMIZADA
- Corrección del problema de detección de polaridad
- Lexicón robusto en español
- Análisis mejorado de contexto
- Manejo correcto de negaciones e intensificadores
"""

import spacy
from textblob import TextBlob
import es_core_news_sm
from collections import defaultdict
import re
from typing import Dict, List, Tuple
import json

class AspectSentimentAnalyzer:
    """
    Analizador de sentimientos basado en aspectos para reseñas de productos.
    """
    
    def __init__(self, language='es'):
        """Inicializa el analizador"""
        self.nlp = es_core_news_sm.load()
        self.language = language
        
        # Diccionario de aspectos
        self.aspect_keywords = {
            'pantalla': {
                'keywords': ['pantalla', 'display', 'screen', 'resolución', 'brillo', 
                           'táctil', 'píxeles', 'pulgadas', 'visualización', 'colores'],
                'color': '#4CAF50'
            },
            'batería': {
                'keywords': ['batería', 'bateria', 'duración', 'carga', 'autonomía', 
                           'mah', 'energía', 'recarga', 'dura', 'aguanta'],
                'color': '#2196F3'
            },
            'cámara': {
                'keywords': ['cámara', 'camara', 'foto', 'fotos', 'video', 'grabación',
                           'megapíxeles', 'lente', 'zoom', 'imagen'],
                'color': '#FF9800'
            },
            'audio': {
                'keywords': ['audio', 'sonido', 'altavoz', 'altavoces', 'volumen',
                           'auriculares', 'música', 'calidad'],
                'color': '#9C27B0'
            },
            'precio': {
                'keywords': ['precio', 'caro', 'barato', 'coste', 'valor', 'económico',
                           'accesible', 'costoso', 'calidad-precio', 'relación'],
                'color': '#F44336'
            },
            'rendimiento': {
                'keywords': ['rendimiento', 'velocidad', 'rápido', 'lento', 'procesador',
                           'ram', 'memoria', 'fluido', 'lag', 'potencia', 'desempeño', 
                           'esperas', 'aplicaciones'],
                'color': '#00BCD4'
            },
            'diseño': {
                'keywords': ['diseño', 'apariencia', 'estética', 'elegante', 'bonito',
                           'acabado', 'construcción', 'materiales', 'aspecto', 'original'],
                'color': '#E91E63'
            },
            'software': {
                'keywords': ['software', 'sistema', 'android', 'ios', 'aplicaciones',
                           'interfaz', 'actualizaciones', 'versión', 'optimizado'],
                'color': '#673AB7'
            }
        }
        
        # Lexicón de sentimientos robusto
        self.sentiment_words = {
            # Muy positivas (+0.8 a +1.0)
            'excelente': 0.9, 'increíble': 0.9, 'espectacular': 0.9, 'fantástico': 0.9,
            'perfecto': 0.95, 'maravilloso': 0.9, 'impresionante': 0.85, 'sobresaliente': 0.85,
            'brillante': 0.8, 'excepcional': 0.9, 'magnífico': 0.85,
            
            # Positivas (+0.4 a +0.7)
            'bueno': 0.6, 'bien': 0.5, 'genial': 0.7, 'bonito': 0.6, 'agradable': 0.5,
            'elegante': 0.6, 'rápido': 0.6, 'fluido': 0.65, 'claro': 0.5, 'nítido': 0.6,
            'potente': 0.65, 'premium': 0.7, 'vibrante': 0.6, 'aguanta': 0.5,
            'decente': 0.4, 'accesible': 0.5, 'optimizado': 0.6,
            
            # Negativas (-0.4 a -0.7)
            'malo': -0.7, 'pobre': -0.6, 'mediocre': -0.6, 'deficiente': -0.65,
            'opaco': -0.5, 'opaca': -0.5, 'lento': -0.6, 'poco': -0.4, 'apenas': -0.5,
            'decepcionante': -0.7, 'insuficiente': -0.6, 'pobres': -0.6,
            
            # Muy negativas (-0.8 a -1.0)
            'pésimo': -0.9, 'horrible': -0.9, 'terrible': -0.9, 'deplorable': -0.95,
            'desastroso': -0.9, 'inaceptable': -0.85, 'defectuoso': -0.8
        }
        
        # Negaciones
        self.negation_words = ['no', 'nunca', 'jamás', 'tampoco', 'sin', 'nada', 'ni']
        
        # Intensificadores
        self.intensifiers = {
            'muy': 1.5, 'bastante': 1.3, 'poco': 0.5, 'demasiado': 1.6,
            'sumamente': 1.8, 'extremadamente': 2.0, 'super': 1.6,
            'tan': 1.4, 'realmente': 1.3
        }
    
    def _extract_syntactic_features(self, token, doc):
        """Extrae características sintácticas"""
        features = {
            'adjectives': [],
            'verbs': [],
            'adverbs': [],
            'negations': []
        }
        
        # Analizar dependencias
        for child in token.children:
            if child.dep_ in ['amod', 'acomp']:
                features['adjectives'].append(child.text)
                for subchild in child.children:
                    if subchild.text.lower() in self.intensifiers:
                        features['adverbs'].append(subchild.text)
            elif child.dep_ == 'neg':
                features['negations'].append(child.text)
        
        # Buscar en el padre
        if token.head != token:
            if token.head.pos_ == 'VERB':
                features['verbs'].append(token.head.text)
            for child in token.head.children:
                if child.dep_ == 'neg' or child.text.lower() in self.negation_words:
                    features['negations'].append(child.text)
        
        return features
    
    def _calculate_sentiment_score(self, context: str, features: dict) -> Tuple[float, str]:
        """Calcula el score de sentimiento MEJORADO"""
        # Análisis base con TextBlob
        blob = TextBlob(context)
        polarity = blob.sentiment.polarity
        
        # MEJORA 1: Buscar palabras del lexicón en el contexto
        context_lower = context.lower()
        lexicon_boost = 0.0
        
        for word, score in self.sentiment_words.items():
            if word in context_lower:
                lexicon_boost += score
        
        # Combinar polaridad base con lexicón
        polarity = (polarity + lexicon_boost) / 2 if lexicon_boost != 0 else polarity
        
        # MEJORA 2: Aplicar intensificadores
        for adverb in features['adverbs']:
            if adverb.lower() in self.intensifiers:
                polarity *= self.intensifiers[adverb.lower()]
        
        # MEJORA 3: Invertir si hay negación
        if features['negations']:
            polarity *= -1
        
        # Normalizar
        polarity = max(-1, min(1, polarity))
        
        # Clasificar con umbrales ajustados
        if polarity > 0.1:  # Umbral más bajo
            sentiment = 'POSITIVO'
        elif polarity < -0.1:
            sentiment = 'NEGATIVO'
        else:
            sentiment = 'NEUTRO'
        
        return polarity, sentiment
    
    def analyze_sentence(self, sentence: str) -> Dict:
        """Analiza una oración"""
        doc = self.nlp(sentence)
        aspects_data = {}
        
        for token in doc:
            for aspect, data in self.aspect_keywords.items():
                if any(keyword in token.text.lower() or keyword in token.lemma_.lower() 
                       for keyword in data['keywords']):
                    
                    # Extraer características
                    features = self._extract_syntactic_features(token, doc)
                    
                    # Contexto más amplio (oración completa es mejor)
                    context = sentence
                    
                    # Calcular sentimiento
                    polarity, sentiment = self._calculate_sentiment_score(context, features)
                    
                    aspects_data[aspect] = {
                        'sentiment': sentiment,
                        'polarity': polarity,
                        'features': features,
                        'context': context.strip(),
                        'sentence': sentence
                    }
        
        return aspects_data
    
    def analyze_review(self, review: str) -> Dict:
        """Analiza un review completo"""
        doc = self.nlp(review)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        results = defaultdict(lambda: {
            'sentiments': [],
            'polarities': [],
            'features': [],
            'contexts': [],
            'count': 0
        })
        
        for sentence in sentences:
            aspects = self.analyze_sentence(sentence)
            for aspect, data in aspects.items():
                results[aspect]['sentiments'].append(data['sentiment'])
                results[aspect]['polarities'].append(data['polarity'])
                results[aspect]['features'].append(data['features'])
                results[aspect]['contexts'].append(data['context'])
                results[aspect]['count'] += 1
        
        # Estadísticas agregadas
        final_results = {}
        for aspect, data in results.items():
            avg_polarity = sum(data['polarities']) / len(data['polarities'])
            
            sentiment_counts = {
                'POSITIVO': data['sentiments'].count('POSITIVO'),
                'NEGATIVO': data['sentiments'].count('NEGATIVO'),
                'NEUTRO': data['sentiments'].count('NEUTRO')
            }
            dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            
            final_results[aspect] = {
                'sentiment_dominant': dominant_sentiment,
                'sentiment_distribution': sentiment_counts,
                'polarity_avg': round(avg_polarity, 3),
                'polarity_min': round(min(data['polarities']), 3),
                'polarity_max': round(max(data['polarities']), 3),
                'mentions': data['count'],
                'contexts': data['contexts'][:3],
                'all_features': data['features']
            }
        
        return final_results
    
    def analyze_multiple_reviews(self, reviews: List[str]) -> Dict:
        """Analiza múltiples reviews"""
        all_results = []
        aspect_aggregation = defaultdict(lambda: {
            'polarities': [],
            'sentiments': [],
            'total_mentions': 0
        })
        
        for review in reviews:
            result = self.analyze_review(review)
            all_results.append(result)
            
            for aspect, data in result.items():
                aspect_aggregation[aspect]['polarities'].append(data['polarity_avg'])
                aspect_aggregation[aspect]['sentiments'].append(data['sentiment_dominant'])
                aspect_aggregation[aspect]['total_mentions'] += data['mentions']
        
        # Métricas globales
        global_metrics = {}
        for aspect, data in aspect_aggregation.items():
            if data['polarities']:
                global_metrics[aspect] = {
                    'avg_polarity': round(sum(data['polarities']) / len(data['polarities']), 3),
                    'total_mentions': data['total_mentions'],
                    'sentiment_distribution': {
                        'POSITIVO': data['sentiments'].count('POSITIVO'),
                        'NEGATIVO': data['sentiments'].count('NEGATIVO'),
                        'NEUTRO': data['sentiments'].count('NEUTRO')
                    },
                    'review_coverage': round(len(data['polarities']) / len(reviews) * 100, 1)
                }
        
        return {
            'individual_results': all_results,
            'global_metrics': global_metrics,
            'total_reviews': len(reviews)
        }
    
    def generate_report(self, analysis_results: Dict) -> str:
        """Genera reporte"""
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE ANÁLISIS DE SENTIMIENTOS POR ASPECTOS")
        report.append("=" * 80)
        report.append(f"\nTotal de reviews analizados: {analysis_results['total_reviews']}")
        report.append("\n" + "-" * 80)
        
        for aspect, metrics in sorted(analysis_results['global_metrics'].items(),
                                     key=lambda x: x[1]['total_mentions'],
                                     reverse=True):
            report.append(f"\n📊 ASPECTO: {aspect.upper()}")
            report.append(f"   └─ Menciones totales: {metrics['total_mentions']}")
            report.append(f"   └─ Cobertura en reviews: {metrics['review_coverage']}%")
            report.append(f"   └─ Polaridad promedio: {metrics['avg_polarity']:.3f}")
            
            dist = metrics['sentiment_distribution']
            total = sum(dist.values())
            report.append(f"   └─ Distribución de sentimientos:")
            report.append(f"      • Positivo: {dist['POSITIVO']} ({dist['POSITIVO']/total*100:.1f}%)")
            report.append(f"      • Negativo: {dist['NEGATIVO']} ({dist['NEGATIVO']/total*100:.1f}%)")
            report.append(f"      • Neutro: {dist['NEUTRO']} ({dist['NEUTRO']/total*100:.1f}%)")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def export_to_json(self, results: Dict, filename: str = "analysis_results.json"):
        """Exporta a JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ Resultados exportados a {filename}")


# ==============================================================================
# EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    analyzer = AspectSentimentAnalyzer()
    
    reviews = [
        """La pantalla es brillante y de excelente calidad, con una resolución 
        impresionante. Sin embargo, la batería dura muy poco tiempo, apenas llega 
        al mediodía. La cámara toma fotos increíbles incluso con poca luz.""",
        
        """El teléfono es bastante caro para lo que ofrece. No obstante, el diseño 
        es elegante y los materiales se sienten premium. El audio es perfecto, 
        con altavoces potentes y claridad excepcional.""",
        
        """No me gusta nada la pantalla, se ve opaca y los colores son pobres. 
        La batería sí aguanta todo el día sin problemas, eso es muy bueno. 
        El rendimiento es fluido y no presenta lag.""",
        
        """Excelente relación calidad-precio. La cámara es decente para el precio, 
        aunque no es la mejor. El software está bien optimizado y recibe 
        actualizaciones frecuentes. La batería podría ser mejor.""",
        
        """El rendimiento es sumamente rápido, no hay esperas al abrir aplicaciones. 
        La pantalla tiene muy buen brillo y los colores son vibrantes. El diseño 
        no es muy original pero se ve bien. Precio bastante accesible."""
    ]
    
    print("\n🔍 INICIANDO ANÁLISIS DE SENTIMIENTOS POR ASPECTOS...")
    print("=" * 80)
    
    # Análisis individual
    print("\n📝 ANÁLISIS INDIVIDUAL DE REVIEWS:\n")
    for i, review in enumerate(reviews, 1):
        print(f"\n{'─' * 80}")
        print(f"Review #{i}:")
        print(f"{'─' * 80}")
        print(f"{review[:100]}...")
        print(f"\nAspectos detectados:")
        
        result = analyzer.analyze_review(review)
        for aspect, data in result.items():
            emoji = "✅" if data['sentiment_dominant'] == 'POSITIVO' else "❌" if data['sentiment_dominant'] == 'NEGATIVO' else "➖"
            print(f"\n  {emoji} {aspect.upper()}:")
            print(f"     • Sentimiento: {data['sentiment_dominant']}")
            print(f"     • Polaridad: {data['polarity_avg']:.3f}")
            print(f"     • Menciones: {data['mentions']}")
            if data['contexts']:
                print(f"     • Contexto: \"{data['contexts'][0][:80]}...\"")
    
    # Análisis agregado
    print("\n\n" + "=" * 80)
    print("📊 ANÁLISIS AGREGADO DE TODOS LOS REVIEWS")
    print("=" * 80)
    
    aggregated_results = analyzer.analyze_multiple_reviews(reviews)
    report = analyzer.generate_report(aggregated_results)
    print(report)
    
    # Exportar
    print("\n💾 Exportando resultados...")
    analyzer.export_to_json(aggregated_results)
    
    print("\n✨ Análisis completado exitosamente!")
    print("=" * 80)