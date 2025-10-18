#!/usr/bin/env python3
"""
Gerador de Timeline Visual do Projeto TCC
Cria visualização do progresso e próximos passos para apresentação ao orientador.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import numpy as np

def create_project_timeline():
    """Cria timeline visual do projeto."""
    
    # Configuração da figura
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle('Timeline do Projeto: Sistema Ensemble FieldSAFE RGB\n' + 
                 'Detecção de Obstáculos Agrícolas', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Definir fases do projeto
    phases = [
        {
            'name': 'FASE 1: Configuração Ambiente',
            'start': 0, 'duration': 1,
            'status': 'completed',
            'details': ['Ambiente virtual Python 3.12', 'PyTorch 2.9.0 + OpenCV', 'Resolução PEP 668']
        },
        {
            'name': 'FASE 2: Pré-processamento',
            'start': 1, 'duration': 2,
            'status': 'completed', 
            'details': ['8.185 imagens processadas', 'Divisão estratificada 65/30/5%', 'Preservação sequências temporais']
        },
        {
            'name': 'FASE 3: Modelos Individuais',
            'start': 3, 'duration': 3,
            'status': 'completed',
            'details': ['YOLOv5 (detecção objetos)', 'SegNet (segmentação)', 'Autoencoder (anomalias)']
        },
        {
            'name': 'FASE 4: Fusão Supervisionada',
            'start': 6, 'duration': 2,
            'status': 'completed',
            'details': ['Multi-head attention (4×)', 'Pesos adaptativos', 'Interface unificada']
        },
        {
            'name': 'FASE 5: Pipeline & Validação',
            'start': 8, 'duration': 1,
            'status': 'completed',
            'details': ['FieldSAFEDataset', 'Testes integração 100%', 'Validação end-to-end']
        },
        {
            'name': 'FASE 6: Treinamento Individual',
            'start': 9, 'duration': 2,
            'status': 'next',
            'details': ['Fine-tuning YOLOv5', 'Treino SegNet', 'Treino Autoencoder']
        },
        {
            'name': 'FASE 7: Fusão Supervisionada',
            'start': 11, 'duration': 1,
            'status': 'future',
            'details': ['Coleta outputs probabilísticos', 'Treino camada fusão', 'Otimização ensemble']
        },
        {
            'name': 'FASE 8: Avaliação Final',
            'start': 12, 'duration': 1,
            'status': 'future',
            'details': ['Teste conjunto reservado', 'Métricas performance', 'Comparação modelos']
        },
        {
            'name': 'FASE 9: Documentação TCC',
            'start': 13, 'duration': 2,
            'status': 'future',
            'details': ['Redação final', 'Resultados experimentais', 'Preparação defesa']
        }
    ]
    
    # Cores para cada status
    colors = {
        'completed': '#2E8B57',  # Verde escuro
        'next': '#FF6B35',       # Laranja
        'future': '#4A90E2'      # Azul
    }
    
    # Desenhar as fases
    y_positions = np.arange(len(phases))[::-1]  # Inverte para mostrar cronologicamente
    
    for i, phase in enumerate(phases):
        y = y_positions[i]
        
        # Barra da fase
        bar = ax.barh(y, phase['duration'], left=phase['start'], 
                     height=0.6, color=colors[phase['status']], 
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        # Nome da fase
        ax.text(phase['start'] + phase['duration']/2, y, 
               phase['name'], 
               ha='center', va='center', fontweight='bold', fontsize=10,
               color='white' if phase['status'] != 'future' else 'black')
        
        # Detalhes da fase (à direita)
        details_text = ' • '.join(phase['details'])
        ax.text(16, y, details_text, ha='left', va='center', fontsize=8,
               style='italic', wrap=True)
    
    # Linha vertical indicando "HOJE"
    today_pos = 9  # Posição atual (após fase 5)
    ax.axvline(x=today_pos, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(today_pos, len(phases), 'HOJE\n(18/10/2025)', ha='center', va='bottom',
           fontweight='bold', color='red', fontsize=12)
    
    # Configurações do gráfico
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])  # Remove labels do eixo Y
    ax.set_xlabel('Semanas do Projeto', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 25)
    ax.set_ylim(-0.5, len(phases) - 0.5)
    
    # Grid suave
    ax.grid(True, axis='x', alpha=0.3, linestyle='-')
    
    # Legenda
    completed_patch = mpatches.Patch(color=colors['completed'], label='✅ CONCLUÍDO')
    next_patch = mpatches.Patch(color=colors['next'], label='🔄 PRÓXIMO PASSO')
    future_patch = mpatches.Patch(color=colors['future'], label='⏳ PLANEJADO')
    
    ax.legend(handles=[completed_patch, next_patch, future_patch], 
             loc='upper right', bbox_to_anchor=(1, 1))
    
    # Estatísticas do progresso
    completed_phases = len([p for p in phases if p['status'] == 'completed'])
    total_phases = len(phases)
    progress_percent = (completed_phases / total_phases) * 100
    
    # Caixa de estatísticas
    stats_text = f"""PROGRESSO ATUAL:
✅ {completed_phases}/{total_phases} fases concluídas ({progress_percent:.1f}%)
📊 8.185 imagens pré-processadas
🤖 3 modelos implementados
🔗 1 camada de fusão pronta
📈 Pipeline 100% validado"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Próximos marcos
    next_milestones = """PRÓXIMOS MARCOS:
🎯 Semana 11-12: Treinamento modelos
🎯 Semana 13: Fusão supervisionada  
🎯 Semana 14: Avaliação ensemble
🎯 Semana 15-16: Documentação final
📝 Estimativa conclusão: 6 semanas"""
    
    ax.text(0.02, 0.02, next_milestones, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_architecture_diagram():
    """Cria diagrama da arquitetura implementada."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle('Arquitetura do Sistema Ensemble Implementado', 
                fontsize=16, fontweight='bold')
    
    # Coordenadas dos componentes
    components = {
        'input': {'pos': (7, 9), 'size': (3, 1), 'label': 'Imagem RGB\n640×640×3'},
        'preprocessing': {'pos': (7, 7.5), 'size': (3, 0.8), 'label': 'Pré-processamento\n(Normalização)'},
        'yolo': {'pos': (2, 5.5), 'size': (2.5, 1.5), 'label': 'YOLOv5\nDetecção Objetos\n5 classes'},
        'segnet': {'pos': (6.25, 5.5), 'size': (2.5, 1.5), 'label': 'SegNet\nSegmentação\n5 classes'},
        'autoencoder': {'pos': (10.5, 5.5), 'size': (2.5, 1.5), 'label': 'Autoencoder\nAnomalias\n2 classes'},
        'fusion': {'pos': (6.25, 3), 'size': (4, 1.5), 'label': 'Fusão Supervisionada\nMulti-Head Attention (4×)\nPesos Adaptativos'},
        'output': {'pos': (7, 1), 'size': (3, 1), 'label': 'Saída Final\nObstáculo/Não-Obstáculo'}
    }
    
    # Cores dos componentes
    component_colors = {
        'input': '#E8F4FD',
        'preprocessing': '#B3D9FF', 
        'yolo': '#FFE6CC',
        'segnet': '#E6F3E6',
        'autoencoder': '#FFE6F0',
        'fusion': '#F0E6FF',
        'output': '#E6FFE6'
    }
    
    # Desenhar componentes
    for name, comp in components.items():
        x, y = comp['pos']
        w, h = comp['size']
        
        # Retângulo do componente
        rect = plt.Rectangle((x - w/2, y - h/2), w, h, 
                           facecolor=component_colors[name],
                           edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Texto do componente
        ax.text(x, y, comp['label'], ha='center', va='center',
               fontsize=10, fontweight='bold', wrap=True)
    
    # Desenhar conexões
    connections = [
        ('input', 'preprocessing'),
        ('preprocessing', 'yolo'),
        ('preprocessing', 'segnet'), 
        ('preprocessing', 'autoencoder'),
        ('yolo', 'fusion'),
        ('segnet', 'fusion'),
        ('autoencoder', 'fusion'),
        ('fusion', 'output')
    ]
    
    for start, end in connections:
        start_pos = components[start]['pos']
        end_pos = components[end]['pos']
        
        # Seta entre componentes
        ax.annotate('', xy=end_pos, xytext=start_pos,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Adicionar anotações de status
    status_annotations = [
        {'pos': (1, 8.5), 'text': '✅ IMPLEMENTADO\nE VALIDADO', 'color': 'green'},
        {'pos': (14, 6), 'text': '📊 DATASET:\n8.185 imagens\n8 sequências\n100% processado', 'color': 'blue'},
        {'pos': (14, 3), 'text': '🎯 PRÓXIMO:\nTreinamento\ndos modelos', 'color': 'orange'}
    ]
    
    for ann in status_annotations:
        ax.text(ann['pos'][0], ann['pos'][1], ann['text'], 
               fontsize=10, fontweight='bold', color=ann['color'],
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlim(-1, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """Gera visualizações do projeto."""
    
    print("🎨 Gerando visualizações para apresentação ao orientador...")
    
    # Timeline do projeto
    timeline_fig = create_project_timeline()
    timeline_fig.savefig('docs/timeline_projeto.png', dpi=300, bbox_inches='tight')
    print("✅ Timeline salva em: docs/timeline_projeto.png")
    
    # Diagrama da arquitetura
    arch_fig = create_architecture_diagram()
    arch_fig.savefig('docs/arquitetura_sistema.png', dpi=300, bbox_inches='tight')
    print("✅ Arquitetura salva em: docs/arquitetura_sistema.png")
    
    # Mostrar plots
    plt.show()
    
    print("\n📋 Arquivos gerados para apresentação:")
    print("   1. docs/relatorio_progresso_orientador.md")
    print("   2. docs/timeline_projeto.png")
    print("   3. docs/arquitetura_sistema.png")
    print("\n🎯 Material completo pronto para apresentação ao orientador!")

if __name__ == "__main__":
    main()