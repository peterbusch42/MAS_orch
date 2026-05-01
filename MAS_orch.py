from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator
import json


# State Definition - zentraler Speicherort für alle Agenten
class ResearchState(TypedDict):
    """
    Zentraler State - ALLE Agenten lesen und schreiben hierhin.
    
    Wichtig: Annotated[List, operator.add] bedeutet:
    - Jeder Agent APPENDED zum State
    - Kein Agent ÜBERSCHREIBT anderen Agent's Output
    - Deterministische State-History
    """
    # Input
    research_question: str
    
    # Kommunikation zwischen Agenten
    messages: Annotated[List[dict], operator.add]
    
    # Task-Tracking
    subtasks: List[str]           # Orchestrator definiert diese
    completed_subtasks: Annotated[List[str], operator.add]
    
    # Ergebnisse (append-only, jeder Agent fügt hinzu)
    research_results: Annotated[List[dict], operator.add]
    
    # Routing-Information
    next_agent: str               # Orchestrator bestimmt, wer als nächstes ran kommt
    
    # Quality Control
    quality_score: Optional[float]
    needs_revision: bool
    
    # Final Output
    final_report: str


# Der Orchestrator Agent steuert den gesamten Research-Prozess.
class OrchestratorAgent:
    """
    Der Orchestrator kennt das Big Picture.
    Er PLANT, DELEGIERT und EVALUIERT - führt selbst keine Research durch.
    
    Kernverantwortlichkeiten:
    1. Task Decomposition (große Aufgabe → kleine Subtasks)
    2. Routing (welcher spezialisierte Agent übernimmt was)
    3. Synthesis (Ergebnisse zusammenführen)
    """
    
    def __init__(self, model: str = "llama3.1:8b"):
        self.llm = ChatOllama(model=model, temperature=0)
        
    def plan_and_route(self, state: ResearchState) -> ResearchState:
        """
        Phase 1: Task Decomposition
        Der Orchestrator zerlegt die Research-Question in Subtasks.
        """
        
        # Wenn noch keine Subtasks existieren → Planning Phase
        if not state.get("subtasks"):
            planning_prompt = f"""
            Du bist ein Research Orchestrator. Deine Aufgabe: Zerlege die folgende 
            Research-Frage in 3-4 konkrete Subtasks, die von spezialisierten Agenten 
            bearbeitet werden können.
            
            Research-Frage: {state['research_question']}
            
            Verfügbare spezialisierte Agenten:
            - "web_researcher": Recherchiert aktuelle Informationen und Fakten
            - "data_analyzer": Analysiert und interpretiert Daten/Statistiken  
            - "synthesizer": Fasst alle Ergebnisse zu einem kohärenten Report zusammen
            
            Antworte NUR mit validem JSON:
            {{
                "subtasks": ["task1", "task2", "task3"],
                "routing_plan": [
                    {{"task": "task1", "agent": "web_researcher"}},
                    {{"task": "task2", "agent": "data_analyzer"}},
                    {{"task": "task3", "agent": "synthesizer"}}
                ],
                "next_agent": "web_researcher"
            }}
            """
            
            response = self.llm.invoke([HumanMessage(content=planning_prompt)])
            plan = json.loads(response.content)
            
            # State Update mit Plan
            return {
                "subtasks": plan["subtasks"],
                "next_agent": plan["next_agent"],
                "messages": [{
                    "from": "orchestrator",
                    "to": plan["next_agent"],
                    "content": f"Bitte bearbeite: {plan['subtasks'][0]}",
                    "context": plan["routing_plan"]
                }]
            }
        
        # Phase 2: Routing nach Ergebnissen der Agenten
        else:
            completed = state.get("completed_subtasks", [])
            all_subtasks = state["subtasks"]
            
            # Alle Tasks erledigt? → Zum Synthesizer
            if len(completed) >= len(all_subtasks):
                return {
                    "next_agent": "synthesizer",
                    "messages": [{
                        "from": "orchestrator",
                        "to": "synthesizer",
                        "content": "Alle Research-Ergebnisse sind bereit. Bitte synthesize.",
                        "results_count": len(state.get("research_results", []))
                    }]
                }
            
            # Nächste ausstehende Task bestimmen
            pending = [t for t in all_subtasks if t not in completed]
            next_task = pending[0]
            
            # Agent für nächste Task bestimmen (simple Heuristik)
            next_agent = "data_analyzer" if "Daten" in next_task or "Statistik" in next_task \
                        else "web_researcher"
            
            return {
                "next_agent": next_agent,
                "messages": [{
                    "from": "orchestrator",
                    "to": next_agent,
                    "content": f"Nächste Aufgabe: {next_task}"
                }]
            }
        

# Spezialisierte Worker-Agenten - führen die eigentliche Research-Arbeit durch
class WebResearcherAgent:
    """
    Spezialisiert auf: Information Retrieval & Faktenrecherche
    
    In Production würde hier ein echter Tool-Call stattfinden:
    - Tavily Search API
    - SerpAPI  
    - Browser-Use
    
    Für das Beispiel: Simulated research via LLM
    """
    
    def __init__(self, model: str = "llama3.1:8b"):
        # Worker-Agenten können günstigere Modelle nutzen!
        self.llm = ChatOllama(model=model, temperature=0.1)
    
    def research(self, state: ResearchState) -> ResearchState:
        """
        Empfängt Task vom Orchestrator über State.
        Führt Research durch.
        Schreibt Ergebnisse zurück in State.
        """
        
        # Aktuellen Task aus Messages extrahieren
        # (In echtem System: strukturiertes Message-Parsing)
        latest_message = state["messages"][-1] if state["messages"] else {}
        current_task = latest_message.get("content", state["research_question"])
        
        research_prompt = f"""
        Du bist ein spezialisierter Web-Research-Agent.
        
        Deine aktuelle Aufgabe: {current_task}
        Übergeordnete Research-Frage: {state['research_question']}
        
        Führe eine detaillierte Recherche durch und liefere:
        1. Konkrete Fakten und Informationen
        2. Quellenangaben (auch wenn simuliert)
        3. Relevanz für die Hauptfrage
        
        Antworte als JSON:
        {{
            "findings": ["finding1", "finding2", "finding3"],
            "key_facts": {{"fact1": "value1", "fact2": "value2"}},
            "sources": ["source1", "source2"],
            "task_completed": "{current_task}",
            "confidence": 0.85
        }}
        """
        
        response = self.llm.invoke([
            SystemMessage(content="Du bist ein präziser Research-Agent. Antworte nur mit validem JSON."),
            HumanMessage(content=research_prompt)
        ])
        
        result = json.loads(response.content)
        
        # Zurück an Orchestrator signalisieren
        return {
            "research_results": [{
                "agent": "web_researcher",
                "task": current_task,
                **result
            }],
            "completed_subtasks": [result["task_completed"]],
            "messages": [{
                "from": "web_researcher",
                "to": "orchestrator",
                "content": f"Research abgeschlossen. {len(result['findings'])} Findings gefunden.",
                "status": "completed"
            }],
            "next_agent": "orchestrator"  # Kontrolle zurück an Orchestrator!
        }


class DataAnalyzerAgent:
    """
    Spezialisiert auf: Quantitative Analyse, Pattern Recognition, Statistiken
    """
    
    def __init__(self, model: str = "llama3.1:8b"):
        # Analyzer braucht mehr Reasoning → stärkeres Modell
        self.llm = ChatOllama(model=model, temperature=0)
    
    def analyze(self, state: ResearchState) -> ResearchState:
        # Alle bisherigen Research-Ergebnisse als Kontext nutzen
        previous_results = state.get("research_results", [])
        context = json.dumps(previous_results, ensure_ascii=False, indent=2)
        
        latest_message = state["messages"][-1] if state["messages"] else {}
        current_task = latest_message.get("content", "Analysiere die vorhandenen Daten")
        
        analysis_prompt = f"""
        Du bist ein Data-Analysis-Agent.
        
        Aufgabe: {current_task}
        
        Bisherige Research-Ergebnisse anderer Agenten:
        {context}
        
        Analysiere die Daten und identifiziere:
        1. Signifikante Patterns und Trends
        2. Quantitative Zusammenhänge
        3. Widersprüche oder Datenlücken
        4. Statistische Relevanz der Findings
        
        JSON-Response:
        {{
            "patterns": ["pattern1", "pattern2"],
            "quantitative_insights": {{"metric1": "value1"}},
            "data_quality": "high|medium|low",
            "gaps_identified": ["gap1"],
            "task_completed": "<aktuelle Task>",
            "confidence": 0.9
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
        result = json.loads(response.content)
        
        return {
            "research_results": [{
                "agent": "data_analyzer",
                "task": current_task,
                **result
            }],
            "completed_subtasks": [result["task_completed"]],
            "messages": [{
                "from": "data_analyzer",
                "to": "orchestrator",
                "content": f"Analyse abgeschlossen. Datenqualität: {result['data_quality']}",
                "status": "completed"
            }],
            "next_agent": "orchestrator"
        }


class SynthesizerAgent:
    """
    Spezialisiert auf: Integration aller Ergebnisse → kohärenter Final Report
    Wird IMMER als letzter Agent ausgeführt.
    """
    
    def __init__(self, model: str = "llama3.1:8b"):
        self.llm = ChatOllama(model=model, temperature=0.3)
    
    def synthesize(self, state: ResearchState) -> ResearchState:
        all_results = state.get("research_results", [])
        
        synthesis_prompt = f"""
        Du bist ein Synthesis-Agent. Deine Aufgabe: Erstelle einen professionellen,
        kohärenten Research-Report.
        
        Ursprüngliche Research-Frage: {state['research_question']}
        
        Alle gesammelten Ergebnisse:
        {json.dumps(all_results, ensure_ascii=False, indent=2)}
        
        Erstelle einen strukturierten Report mit:
        1. Executive Summary (3-5 Sätze)
        2. Haupterkenntnisse (priorisiert)
        3. Datenbasierte Schlussfolgerungen
        4. Offene Fragen / Empfehlungen für weitere Recherche
        5. Confidence-Score des Gesamtergebnisses
        
        Schreibe professionell, präzise und evidenzbasiert.
        """
        
        response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
        
        return {
            "final_report": response.content,
            "messages": [{
                "from": "synthesizer",
                "to": "orchestrator",
                "content": "Final Report erstellt.",
                "status": "final"
            }],
            "next_agent": "quality_checker"
        }


class QualityCheckerAgent:
    """
    Optional: Validierungsschicht
    Überprüft ob der Report die Research-Frage tatsächlich beantwortet.
    """
    
    def __init__(self, model: str = "llama3.1:8b"):
        self.llm = ChatOllama(model=model, temperature=0)
    
    def check(self, state: ResearchState) -> ResearchState:
        check_prompt = f"""
        Bewerte den folgenden Research-Report:
        
        Ursprüngliche Frage: {state['research_question']}
        
        Report:
        {state.get('final_report', '')}
        
        Bewertungskriterien:
        - Beantwortet der Report die Frage? (0-1)
        - Ist er evidenzbasiert? (0-1)
        - Gibt es kritische Lücken? (true/false)
        
        JSON:
        {{
            "answers_question": 0.9,
            "is_evidence_based": 0.85,
            "has_critical_gaps": false,
            "overall_score": 0.87,
            "needs_revision": false,
            "revision_reason": ""
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=check_prompt)])
        result = json.loads(response.content)
        
        return {
            "quality_score": result["overall_score"],
            "needs_revision": result["needs_revision"],
            "messages": [{
                "from": "quality_checker",
                "content": f"Quality Score: {result['overall_score']}",
                "needs_revision": result["needs_revision"]
            }],
            "next_agent": "orchestrator" if result["needs_revision"] else END
        }
    

#Graph Construction: Hier definieren wir, wie die Agenten miteinander verbunden sind und wie der Flow durch das System läuft.
def create_research_graph() -> StateGraph:
    """
    Baut den LangGraph-Graphen.
    
    Kritisches Konzept: 
    - Nodes = Agenten (was wird ausgeführt)
    - Edges = Routing (wer kommt als nächstes)
    - Conditional Edges = Dynamisches Routing basierend auf State
    """
    
    # Agent-Instanzen
    orchestrator = OrchestratorAgent()
    researcher = WebResearcherAgent()
    analyzer = DataAnalyzerAgent()
    synthesizer = SynthesizerAgent()
    quality_checker = QualityCheckerAgent()
    
    # Graph initialisieren
    graph = StateGraph(ResearchState)
    
    # ── NODES HINZUFÜGEN ────────────────────────────────────────────
    graph.add_node("orchestrator", orchestrator.plan_and_route)
    graph.add_node("web_researcher", researcher.research)
    graph.add_node("data_analyzer", analyzer.analyze)
    graph.add_node("synthesizer", synthesizer.synthesize)
    graph.add_node("quality_checker", quality_checker.check)
    
    # ── ENTRY POINT ─────────────────────────────────────────────────
    graph.set_entry_point("orchestrator")
    
    # ── CONDITIONAL EDGES (Das Herzstück des Routings) ───────────────
    def route_from_orchestrator(state: ResearchState) -> str:
        """
        Diese Funktion bestimmt nach jedem Orchestrator-Call:
        Wohin geht der Flow als nächstes?
        
        Basiert NUR auf State - kein direkter Agent-zu-Agent-Call!
        """
        next_agent = state.get("next_agent", "web_researcher")
        
        # Quality-Check: Revision nötig?
        if state.get("needs_revision"):
            return "web_researcher"  # Re-research triggern
        
        valid_routes = {
            "web_researcher": "web_researcher",
            "data_analyzer": "data_analyzer", 
            "synthesizer": "synthesizer",
            "quality_checker": "quality_checker",
        }
        
        return valid_routes.get(next_agent, "synthesizer")
    
    def route_after_quality_check(state: ResearchState) -> str:
        """Terminierung oder Revision?"""
        if state.get("needs_revision", False):
            return "orchestrator"  # Zurück zum Start → Revision Loop
        return END
    
    # Orchestrator kann zu ALLEN Agenten routen
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "web_researcher": "web_researcher",
            "data_analyzer": "data_analyzer",
            "synthesizer": "synthesizer",
            "quality_checker": "quality_checker",
        }
    )
    
    # Worker-Agenten geben Kontrolle IMMER zurück an Orchestrator
    graph.add_edge("web_researcher", "orchestrator")
    graph.add_edge("data_analyzer", "orchestrator")
    graph.add_edge("synthesizer", "quality_checker")  # Synthesizer → direkt zu QC
    
    # Quality-Check: Terminierung oder Revision
    graph.add_conditional_edges(
        "quality_checker",
        route_after_quality_check,
        {
            "orchestrator": "orchestrator",
            END: END
        }
    )
    
    return graph.compile()


# Main Execution Loop: Hier wird das System tatsächlich ausgeführt.
def run_research_system(question: str):
    """
    Hauptfunktion: Startet das Multi-Agenten-System.
    """
    
    # Graph kompilieren
    app = create_research_graph()
    
    # Initial State
    initial_state: ResearchState = {
        "research_question": question,
        "messages": [],
        "subtasks": [],
        "completed_subtasks": [],
        "research_results": [],
        "next_agent": "orchestrator",
        "quality_score": None,
        "needs_revision": False,
        "final_report": ""
    }
    
    print(f"🚀 Starte Research-System für: {question}\n")
    print("=" * 60)
    
    # Stream-Execution: Jeden Step live beobachten
    # Das ist KRITISCH für Debugging und Monitoring!
    for step in app.stream(initial_state, config={"recursion_limit": 20}):
        
        for node_name, node_output in step.items():
            print(f"\n📍 Agent: {node_name.upper()}")
            print(f"   Next: {node_output.get('next_agent', 'N/A')}")
            
            # Messages anzeigen
            new_messages = node_output.get("messages", [])
            for msg in new_messages:
                print(f"   💬 [{msg.get('from', '?')} → {msg.get('to', '?')}]: {msg.get('content', '')[:100]}")
            
            # Neue Results
            new_results = node_output.get("research_results", [])
            if new_results:
                print(f"   📊 Neue Findings: {len(new_results)} Ergebnis(se)")
            
            # Completed Tasks
            completed = node_output.get("completed_subtasks", [])
            if completed:
                print(f"   ✅ Abgeschlossen: {completed}")
    
    # Finale Ergebnisse aus komplettem State holen
    final_state = app.invoke(initial_state, config={"recursion_limit": 20})
    
    print("\n" + "=" * 60)
    print("📋 FINAL REPORT:")
    print("=" * 60)
    print(final_state.get("final_report", "Kein Report generiert"))
    print(f"\n⭐ Quality Score: {final_state.get('quality_score', 'N/A')}")
    
    return final_state


# Ausführung
if __name__ == "__main__":
    result = run_research_system(
        "Welche Auswirkungen hat Quantum Computing auf aktuelle Kryptographie-Standards?"
    )

