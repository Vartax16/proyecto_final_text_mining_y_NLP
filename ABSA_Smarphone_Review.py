# ============================================================================
# INSTALACIÓN DE DEPENDENCIAS
# ============================================================================

# CELDA 1: Instalar librerías
!pip install spacy textblob transformers torch pandas numpy matplotlib seaborn scikit-learn -q
!python -m spacy download es_core_news_sm -q
!python -m textblob.download_corpora

print("✓ Librerías instaladas correctamente")


# ============================================================================
# IMPORTACIONES
# ============================================================================

# CELDA 2: Importar librerías

import spacy
from textblob import TextBlob
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re
from typing import Dict, List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Para visualizaciones
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("✓ Importaciones completadas")


# ============================================================================
# CLASE PRINCIPAL: ANALIZADOR ABSA
# ============================================================================

# CELDA 3: Definicion clase AspectSentimentAnalyzer

class AspectSentimentAnalyzer:
    """
    Analizador Aspect-Based Sentiment Analysis (ABSA) para reviews de smartphones.
    Diseñado específicamente para el dataset phone_reviews.csv
    """
    
    def __init__(self, language='en'):
        """Inicializa el analizador"""
        print("🔄 Cargando modelo spaCy...")
        self.nlp = spacy.load('es_core_news_sm')
        self.language = language
        
        # ====== DICCIONARIO DE ASPECTOS PARA SMARTPHONES ======
        self.aspect_keywords = {
            'pantalla': {
                'keywords': ['pantalla', 'display', 'screen', 'resolución', 'brillo', 
                           'táctil', 'píxeles', 'pulgadas', 'visualización', 'colores',
                           'lcd', 'oled', 'retina', 'ips', 'fhd', 'qhd', 'vista'],
                'color': '#4CAF50',
                'emoji': '📱'
            },
            'batería': {
                'keywords': ['batería', 'bateria', 'duración', 'carga', 'autonomía', 
                           'mah', 'energía', 'recarga', 'dura', 'aguanta', 'consumo',
                           'rápida', 'rápido', 'larga', 'corta'],
                'color': '#2196F3',
                'emoji': '🔋'
            },
            'cámara': {
                'keywords': ['cámara', 'camara', 'foto', 'fotos', 'video', 'grabación',
                           'megapíxeles', 'lente', 'zoom', 'imagen', 'fotografía',
                           'nocturna', 'noche', 'retrato', 'ultra-gran'],
                'color': '#FF9800',
                'emoji': '📷'
            },
            'audio': {
                'keywords': ['audio', 'sonido', 'altavoz', 'altavoces', 'volumen',
                           'auriculares', 'música', 'calidad', 'estéreo', 'graves',
                           'agudos', 'jack', 'altoparlantes'],
                'color': '#9C27B0',
                'emoji': '🔊'
            },
            'precio': {
                'keywords': ['precio', 'caro', 'barato', 'coste', 'valor', 'económico',
                           'accesible', 'costoso', 'calidad-precio', 'relación',
                           'inversión', 'oferta', 'descuento'],
                'color': '#F44336',
                'emoji': '💰'
            },
            'rendimiento': {
                'keywords': ['rendimiento', 'velocidad', 'rápido', 'lento', 'procesador',
                           'ram', 'memoria', 'fluido', 'lag', 'potencia', 'desempeño', 
                           'esperas', 'aplicaciones', 'juegos', 'snapdragon', 'exynos'],
                'color': '#00BCD4',
                'emoji': '⚡'
            },
            'diseño': {
                'keywords': ['diseño', 'apariencia', 'estética', 'elegante', 'bonito',
                           'acabado', 'construcción', 'materiales', 'aspecto', 'original',
                           'peso', 'grosor', 'aluminio', 'vidrio', 'plástico'],
                'color': '#E91E63',
                'emoji': '🎨'
            },
            'software': {
                'keywords': ['software', 'sistema', 'android', 'ios', 'aplicaciones',
                           'interfaz', 'actualizaciones', 'versión', 'optimizado',
                           'miui', 'oneui', 'oxygenos', 'fluidosidad'],
                'color': '#673AB7',
                'emoji': '💻'
            }
        }
        
        # ====== LEXICÓN DE SENTIMIENTOS ======
        self.sentiment_words = {
            # Muy positivas (+0.8 a +1.0)
            'excelente': 0.9, 'increíble': 0.9, 'espectacular': 0.9, 'fantástico': 0.9,
            'perfecto': 0.95, 'maravilloso': 0.9, 'impresionante': 0.85, 'sobresaliente': 0.85,
            'brillante': 0.8, 'excepcional': 0.9, 'magnífico': 0.85, 'superior': 0.8,
            'outstanding': 0.9, 'excellent': 0.9, 'amazing': 0.9, 'wonderful': 0.85,
            'perfect': 0.95, 'impressive': 0.85, 'great': 0.75,
            
            # Positivas (+0.4 a +0.7)
            'bueno': 0.6, 'bien': 0.5, 'genial': 0.7, 'bonito': 0.6, 'agradable': 0.5,
            'elegante': 0.6, 'rápido': 0.6, 'fluido': 0.65, 'claro': 0.5, 'nítido': 0.6,
            'potente': 0.65, 'premium': 0.7, 'vibrante': 0.6, 'aguanta': 0.5,
            'decente': 0.4, 'accesible': 0.5, 'optimizado': 0.6, 'satisfecho': 0.65,
            'good': 0.6, 'nice': 0.5, 'cool': 0.6, 'smooth': 0.65, 'sharp': 0.6,
            'powerful': 0.65, 'fast': 0.6, 'clear': 0.5,
            
            # Negativas (-0.4 a -0.7)
            'malo': -0.7, 'pobre': -0.6, 'mediocre': -0.6, 'deficiente': -0.65,
            'opaco': -0.5, 'opaca': -0.5, 'lento': -0.6, 'poco': -0.4, 'apenas': -0.5,
            'decepcionante': -0.7, 'insuficiente': -0.6, 'pobres': -0.6, 'feo': -0.65,
            'incómodo': -0.6, 'difícil': -0.5, 'inestable': -0.65,
            'bad': -0.7, 'poor': -0.6, 'slow': -0.6, 'difficult': -0.5,
            
            # Muy negativas (-0.8 a -1.0)
            'pésimo': -0.9, 'horrible': -0.9, 'terrible': -0.9, 'deplorable': -0.95,
            'desastroso': -0.9, 'inaceptable': -0.85, 'defectuoso': -0.8, 'roto': -0.85,
            'basura': -0.95, 'peor': -0.8, 'dreadful': -0.9, 'awful': -0.9
        }
        
        # Negaciones
        self.negation_words = ['no', 'nunca', 'jamás', 'tampoco', 'sin', 'nada', 'ni',
                              'not', 'never', 'neither', 'none']
        
        # Intensificadores
        self.intensifiers = {
            'muy': 1.5, 'bastante': 1.3, 'poco': 0.5, 'demasiado': 1.6,
            'sumamente': 1.8, 'extremadamente': 2.0, 'super': 1.6, 'tan': 1.4,
            'realmente': 1.3, 'really': 1.3, 'very': 1.5, 'too': 1.6, 'so': 1.5
        }
        
        print("✓ Modelo cargado correctamente")
    
    def _extract_syntactic_features(self, token, doc):
        """Extrae características sintácticas usando Dependency Parsing"""
        features = {
            'adjectives': [],
            'verbs': [],
            'adverbs': [],
            'negations': []
        }
        
        # Analizar dependencias
        for child in token.children:
            if child.dep_ in ['amod', 'acomp', 'attr']:
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
        try:
            blob = TextBlob(context)
            polarity = blob.sentiment.polarity
        except:
            polarity = 0.0
        
        # Búsqueda en lexicón personalizado
        context_lower = context.lower()
        lexicon_boost = 0.0
        word_count = 0
        
        for word, score in self.sentiment_words.items():
            if word in context_lower:
                lexicon_boost += score
                word_count += 1
        
        # Combinar: 60% lexicón, 40% TextBlob
        if word_count > 0:
            lexicon_avg = lexicon_boost / word_count
            polarity = (0.6 * lexicon_avg + 0.4 * polarity)
        
        # Aplicar intensificadores
        for adverb in features['adverbs']:
            if adverb.lower() in self.intensifiers:
                multiplier = self.intensifiers[adverb.lower()]
                polarity = polarity * multiplier if polarity > 0 else polarity * (2 - multiplier)
        
        # Invertir si hay negación
        if features['negations']:
            polarity *= -1
        
        # Normalizar
        polarity = max(-1, min(1, polarity))
        
        # Clasificar con umbrales ajustados
        if polarity > 0.15:
            sentiment = 'POSITIVO'
        elif polarity < -0.15:
            sentiment = 'NEGATIVO'
        else:
            sentiment = 'NEUTRO'
        
        return round(polarity, 3), sentiment
    
    def analyze_sentence(self, sentence: str) -> Dict:
        """Analiza una oración para extraer aspectos"""
        try:
            doc = self.nlp(sentence)
        except:
            return {}
        
        aspects_data = {}
        
        for token in doc:
            for aspect, data in self.aspect_keywords.items():
                # Verificar si el token coincide con palabras clave del aspecto
                token_lower = token.text.lower()
                lemma_lower = token.lemma_.lower()
                
                if any(keyword in token_lower or keyword in lemma_lower 
                       for keyword in data['keywords']):
                    
                    # Extraer características
                    features = self._extract_syntactic_features(token, doc)
                    
                    # Contexto completo de la oración
                    context = sentence
                    
                    # Calcular sentimiento
                    polarity, sentiment = self._calculate_sentiment_score(context, features)
                    
                    aspects_data[aspect] = {
                        'sentiment': sentiment,
                        'polarity': polarity,
                        'features': features,
                        'context': context.strip(),
                        'token': token.text
                    }
        
        return aspects_data
    
    def analyze_review(self, review: str) -> Dict:
        """Analiza un review completo (múltiples oraciones)"""
        try:
            doc = self.nlp(review)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except:
            sentences = [review]
        
        results = defaultdict(lambda: {
            'sentiments': [],
            'polarities': [],
            'contexts': [],
            'count': 0
        })
        
        for sentence in sentences:
            aspects = self.analyze_sentence(sentence)
            for aspect, data in aspects.items():
                results[aspect]['sentiments'].append(data['sentiment'])
                results[aspect]['polarities'].append(data['polarity'])
                results[aspect]['contexts'].append(data['context'])
                results[aspect]['count'] += 1
        
        # Estadísticas agregadas por review
        final_results = {}
        for aspect, data in results.items():
            if data['polarities']:
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
                    'contexts': data['contexts'][:2]
                }
        
        return final_results
    
    def analyze_multiple_reviews(self, reviews: List[str], show_progress=True) -> Dict:
        """Analiza múltiples reviews con agregación - ⭐ CORREGIDO"""
        all_results = []
        aspect_aggregation = defaultdict(lambda: {
            'polarities': [],
            'sentiments': [],
            'total_mentions': 0,
            'reviews_with_mention': 0
        })
        
        total = len(reviews)
        for idx, review in enumerate(reviews):
            if show_progress and (idx + 1) % 100 == 0:
                print(f"  ⏳ Procesados {idx + 1}/{total} reviews...")
            
            result = self.analyze_review(review)
            all_results.append(result)
            
            for aspect, data in result.items():
                aspect_aggregation[aspect]['polarities'].append(data['polarity_avg'])
                aspect_aggregation[aspect]['sentiments'].append(data['sentiment_dominant'])
                aspect_aggregation[aspect]['total_mentions'] += data['mentions']
                aspect_aggregation[aspect]['reviews_with_mention'] += 1
        
        # Métricas globales por aspecto - ⭐ LÍNEA CORREGIDA (sentimientos → sentiments)
        global_metrics = {}
        for aspect, data in aspect_aggregation.items():
            if data['polarities']:
                sentiment_dist = {
                    'POSITIVO': data['sentiments'].count('POSITIVO'),
                    'NEGATIVO': data['sentiments'].count('NEGATIVO'),
                    'NEUTRO': data['sentiments'].count('NEUTRO')
                }
                
                global_metrics[aspect] = {
                    'avg_polarity': round(sum(data['polarities']) / len(data['polarities']), 3),
                    'total_mentions': data['total_mentions'],
                    'sentiment_distribution': sentiment_dist,
                    'review_coverage': round(data['reviews_with_mention'] / total * 100, 1),
                    'positive_percentage': round(sentiment_dist['POSITIVO'] / sum(sentiment_dist.values()) * 100, 1) if sum(sentiment_dist.values()) > 0 else 0
                }
        
        return {
            'individual_results': all_results,
            'global_metrics': global_metrics,
            'total_reviews': total,
            'aspects_found': len(global_metrics)
        }
    
    def generate_report(self, analysis_results: Dict) -> str:
        """Genera reporte textual"""
        report = []
        report.append("=" * 90)
        report.append("🔍 REPORTE DE ANÁLISIS ASPECT-BASED SENTIMENT ANALYSIS (ABSA)")
        report.append("=" * 90)
        report.append(f"\n📊 ESTADÍSTICAS GENERALES:")
        report.append(f"   • Total de reviews analizados: {analysis_results['total_reviews']:,}")
        report.append(f"   • Aspectos detectados: {analysis_results['aspects_found']}")
        report.append(f"   • Timestamp: {pd.Timestamp.now()}")
        
        report.append("\n" + "-" * 90)
        report.append("RESULTADOS POR ASPECTO (Ordenados por Cobertura):\n")
        
        # Ordenar por cobertura
        sorted_aspects = sorted(analysis_results['global_metrics'].items(),
                               key=lambda x: x[1]['review_coverage'],
                               reverse=True)
        
        for aspect, metrics in sorted_aspects:
            emoji = self.aspect_keywords[aspect]['emoji']
            polarity = metrics['avg_polarity']
            
            # Emoji de sentimiento
            if polarity > 0.3:
                sentiment_emoji = "✅"
            elif polarity < -0.3:
                sentiment_emoji = "❌"
            else:
                sentiment_emoji = "➖"
            
            report.append(f"\n{emoji} {aspect.upper()}")
            report.append(f"   {sentiment_emoji} Polaridad promedio: {polarity:.3f} (rango: -1 a +1)")
            report.append(f"   📈 Cobertura: {metrics['review_coverage']}% de reviews mencionan este aspecto")
            report.append(f"   💬 Total menciones: {metrics['total_mentions']}")
            report.append(f"   😊 Sentimiento positivo: {metrics['positive_percentage']:.1f}%")
            
            dist = metrics['sentiment_distribution']
            total = sum(dist.values())
            if total > 0:
                report.append(f"   📊 Distribución:")
                report.append(f"      ✅ Positivo: {dist.get('POSITIVO', 0)} ({dist.get('POSITIVO', 0)/total*100:.1f}%)")
                report.append(f"      ❌ Negativo: {dist.get('NEGATIVO', 0)} ({dist.get('NEGATIVO', 0)/total*100:.1f}%)")
                report.append(f"      ➖ Neutro: {dist.get('NEUTRO', 0)} ({dist.get('NEUTRO', 0)/total*100:.1f}%)")
        
        report.append("\n" + "=" * 90)
        return "\n".join([l for l in report if l])  # Filtrar líneas vacías
    
    def export_to_json(self, results: Dict, filename: str = "absa_results.json"):
        """Exporta resultados a JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ Resultados exportados a {filename}")
        return filename
    
    def export_to_csv(self, results: Dict, filename: str = "absa_metrics_summary.csv"):
        """Exporta métricas globales a CSV"""
        data = []
        for aspect, metrics in results['global_metrics'].items():
            data.append({
                'Aspecto': aspect,
                'Polaridad_Promedio': metrics['avg_polarity'],
                'Total_Menciones': metrics['total_mentions'],
                'Cobertura_%': metrics['review_coverage'],
                'Positivo_%': metrics['positive_percentage'],
                'Sentimiento_Dominante': max(metrics['sentiment_distribution'],
                                            key=metrics['sentiment_distribution'].get)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Polaridad_Promedio', ascending=False)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"✓ Resultados exportados a {filename}")
        return df


print("✓ Clase AspectSentimentAnalyzer definida correctamente")


# ============================================================================
# PIPELINE DE ANÁLISIS
# ============================================================================

# CELDA 4: Cargar dataset

# Cargar archivo CSV
df = pd.read_csv('./phone_reviews.csv')

print(f"✓ Dataset cargado: {len(df):,} reviews")
print(f"✓ Columnas: {list(df.columns)}")
print(f"\nPrimeras 3 reviews:")
print(df[['mobile_names', 'body', 'star']].head(3))


# CELDA 5: Preparar datos

# Combinar título y cuerpo
df['full_review'] = df['title'].fillna('') + ' ' + df['body'].fillna('')
df['full_review'] = df['full_review'].str.strip()

# Filtrar reviews vacíos
df = df[df['full_review'].str.len() > 3].reset_index(drop=True)

print(f"✓ Dataset preparado: {len(df):,} reviews válidos")

# Mostrar estadísticas
print(f"\n📊 ESTADÍSTICAS DEL DATASET:")
print(f"  • Dispositivos únicos: {df['mobile_names'].nunique()}")
print(f"  • Rating promedio: {df['star'].mean():.2f}⭐")
print(f"  • Longitud promedio review: {df['full_review'].str.len().mean():.0f} caracteres")


# CELDA 6: Inicializar analizador

analyzer = AspectSentimentAnalyzer(language='en')


# CELDA 7: Ejecutar análisis (puede tomar varios minutos)

print("\n🚀 INICIANDO ANÁLISIS ABSA...")
print("=" * 90)

# OPCIÓN 1: Análisis rápido (primeros 1000 reviews)
df_sample = df.head(1000)

# OPCIÓN 2: Análisis completo (17,248 reviews - toma ~30-120 min)
# df_sample = df

# OPCIÓN 3: Análisis por dispositivo específico
# df_sample = df[df['mobile_names'].str.contains('iPhone')]

print(f"\n🔄 Analizando {len(df_sample):,} reviews...")
reviews = df_sample['full_review'].tolist()

analysis_results = analyzer.analyze_multiple_reviews(reviews, show_progress=True)

print(f"\n✓ Análisis completado exitosamente!")
print(f"✓ Aspectos detectados: {analysis_results['aspects_found']}")


# CELDA 8: Generar reporte

report = analyzer.generate_report(analysis_results)
print("\n" + report)


# ============================================================================
# EXPORTAR RESULTADOS
# ============================================================================

# CELDA 9: Exportar a archivos

print("\n💾 EXPORTANDO RESULTADOS...")

# JSON
json_file = analyzer.export_to_json(analysis_results, 'absa_analysis_results.json')

# CSV
csv_df = analyzer.export_to_csv(analysis_results, 'absa_metrics_summary.csv')


# ============================================================================
# VISUALIZACIONES
# ============================================================================

# CELDA 10: Gráficos principales

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gráfico 1: Polaridad por aspecto
aspects = list(analysis_results['global_metrics'].keys())
polarities = [analysis_results['global_metrics'][a]['avg_polarity'] for a in aspects]
colors = ['#4CAF50' if p > 0 else '#F44336' for p in polarities]

axes[0, 0].barh(aspects, polarities, color=colors)
axes[0, 0].set_xlabel('Polaridad Promedio')
axes[0, 0].set_title('Sentimiento Promedio por Aspecto', fontweight='bold', fontsize=12)
axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[0, 0].set_xlim(-1, 1)

# Gráfico 2: Cobertura de aspectos
coverages = [analysis_results['global_metrics'][a]['review_coverage'] for a in aspects]
axes[0, 1].bar(aspects, coverages, color='#2196F3')
axes[0, 1].set_ylabel('% de Reviews')
axes[0, 1].set_title('Cobertura de Aspectos (%)', fontweight='bold', fontsize=12)
axes[0, 1].set_ylim(0, 100)
plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Gráfico 3: Distribución de sentimientos
sentiments_total = {'POSITIVO': 0, 'NEGATIVO': 0, 'NEUTRO': 0}
for a in aspects:
    dist = analysis_results['global_metrics'][a]['sentiment_distribution']
    for sentiment, count in dist.items():
        sentiments_total[sentiment] += count

colors_sentiments = ['#4CAF50', '#F44336', '#FFC107']
axes[1, 0].pie(sentiments_total.values(), labels=sentiments_total.keys(), 
               autopct='%1.1f%%', colors=colors_sentiments, startangle=90)
axes[1, 0].set_title('Distribución Global de Sentimientos', fontweight='bold', fontsize=12)

# Gráfico 4: Menciones por aspecto
mentions = [analysis_results['global_metrics'][a]['total_mentions'] for a in aspects]
axes[1, 1].bar(aspects, mentions, color='#9C27B0')
axes[1, 1].set_ylabel('Total de Menciones')
axes[1, 1].set_title('Frecuencia de Menciones por Aspecto', fontweight='bold', fontsize=12)
plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('absa_analysis_charts.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Gráficos guardados en 'absa_analysis_charts.png'")


# CELDA 11: Análisis por dispositivo (opcional)

print("\n" + "=" * 90)
print("📱 ANÁLISIS POR DISPOSITIVO (TOP 5)")
print("=" * 90)

# Seleccionar top 5 dispositivos
top_devices = df_sample['mobile_names'].value_counts().head(5)

for device in top_devices.index:
    device_reviews = df_sample[df_sample['mobile_names'] == device]['full_review'].tolist()
    
    if len(device_reviews) > 0:
        device_results = analyzer.analyze_multiple_reviews(device_reviews, show_progress=False)
        
        print(f"\n📱 {device}")
        print(f"   • Reviews analizados: {len(device_reviews)}")
        
        if device_results['global_metrics']:
            # Top 3 aspectos positivos
            sorted_by_polarity = sorted(device_results['global_metrics'].items(),
                                       key=lambda x: x[1]['avg_polarity'],
                                       reverse=True)
            
            print(f"   • Mejor aspecto: {sorted_by_polarity[0][0]} ({sorted_by_polarity[0][1]['avg_polarity']:.3f})")
            print(f"   • Peor aspecto: {sorted_by_polarity[-1][0]} ({sorted_by_polarity[-1][1]['avg_polarity']:.3f})")


# CELDA 12: Análisis por rating

print("\n" + "=" * 90)
print("⭐ ANÁLISIS POR RATING")
print("=" * 90)

for rating in sorted(df_sample['star'].unique(), reverse=True):
    rating_reviews = df_sample[df_sample['star'] == rating]['full_review'].tolist()
    
    if len(rating_reviews) > 0:
        rating_results = analyzer.analyze_multiple_reviews(rating_reviews, show_progress=False)
        
        print(f"\n{'⭐' * rating} ({len(rating_reviews)} reviews)")
        
        if rating_results['global_metrics']:
            # Polaridad promedio para este rating
            avg_polarity = np.mean([m['avg_polarity'] for m in rating_results['global_metrics'].values()])
            print(f"   • Polaridad promedio ABSA: {avg_polarity:.3f}")
            
            # Aspecto más mencionado
            most_mentioned = max(rating_results['global_metrics'].items(),
                               key=lambda x: x[1]['total_mentions'])
            print(f"   • Aspecto más mencionado: {most_mentioned[0]} ({most_mentioned[1]['total_mentions']} menciones)")


print("\n✨ Análisis completado exitosamente!")
print("=" * 90)