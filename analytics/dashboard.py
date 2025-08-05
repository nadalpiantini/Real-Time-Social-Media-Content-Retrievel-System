"""
Real-Time Analytics Dashboard
Interactive visualizations for search analytics, content insights, and system performance
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid

from analytics.metrics_collector import get_metrics_collector
from analytics.clustering import get_clustering_engine
from models.post import EmbeddedChunkedPost


class AnalyticsDashboard:
    """Real-time analytics dashboard with interactive visualizations"""
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self.clustering_engine = get_clustering_engine()
        
        # Initialize session state for dashboard
        if 'dashboard_session_id' not in st.session_state:
            st.session_state.dashboard_session_id = str(uuid.uuid4())
        
        if 'last_analytics_update' not in st.session_state:
            st.session_state.last_analytics_update = datetime.now()
    
    def render_dashboard(self):
        """Render the complete analytics dashboard"""
        
        st.header("📊 Real-Time Analytics Dashboard")
        st.markdown("Monitor search patterns, content insights, and system performance in real-time")
        
        # Dashboard controls
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            time_range = st.selectbox(
                "Time Range",
                ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
                index=2
            )
        
        with col2:
            auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
        
        with col3:
            if st.button("🔄 Refresh Now"):
                st.session_state.last_analytics_update = datetime.now()
                st.rerun()
        
        # Convert time range to hours
        time_hours = {
            "Last Hour": 1,
            "Last 6 Hours": 6, 
            "Last 24 Hours": 24,
            "Last 7 Days": 168
        }[time_range]
        
        # Auto-refresh logic
        if auto_refresh:
            time_since_update = datetime.now() - st.session_state.last_analytics_update
            if time_since_update.total_seconds() > 30:
                st.session_state.last_analytics_update = datetime.now()
                st.rerun()
        
        # Get analytics data
        search_analytics = self.metrics_collector.get_search_analytics(time_hours)
        content_analytics = self.metrics_collector.get_content_analytics()
        system_analytics = self.metrics_collector.get_system_analytics()
        
        # Render dashboard sections
        self._render_overview_metrics(search_analytics, content_analytics, system_analytics)
        
        st.markdown("---")
        
        # Create tabs for different analytics views
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Search Analytics", 
            "📚 Content Insights", 
            "🖥️ System Performance",
            "🧩 Semantic Clustering"
        ])
        
        with tab1:
            self._render_search_analytics(search_analytics, time_hours)
        
        with tab2:
            self._render_content_analytics(content_analytics)
        
        with tab3:
            self._render_system_analytics(system_analytics)
        
        with tab4:
            self._render_clustering_analytics()
    
    def _render_overview_metrics(self, search_analytics: Dict, content_analytics: Dict, system_analytics: Dict):
        """Render overview metrics cards"""
        
        st.subheader("📈 Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Searches",
                search_analytics.get("total_searches", 0),
                help="Total number of searches in selected time period"
            )
        
        with col2:
            avg_response = search_analytics.get("avg_response_time_ms", 0)
            st.metric(
                "Avg Response Time",
                f"{avg_response:.0f}ms",
                delta=f"{'🚀' if avg_response < 500 else '🐌' if avg_response > 2000 else '⚡'}",
                help="Average search response time"
            )
        
        with col3:
            st.metric(
                "Content Items",
                content_analytics.get("total_content_items", 0),
                help="Total content items in the system"
            )
        
        with col4:
            active_sessions = search_analytics.get("active_sessions", 0)
            st.metric(
                "Active Sessions",
                active_sessions,
                help="Currently active user sessions"
            )
        
        with col5:
            cache_hit_rate = system_analytics.get("current_cache_hit_rate", 0)
            st.metric(
                "Cache Hit Rate",
                f"{cache_hit_rate:.1f}%",
                delta="✅" if cache_hit_rate > 80 else "⚠️" if cache_hit_rate > 50 else "❌",
                help="Cache efficiency percentage"
            )
    
    def _render_search_analytics(self, analytics: Dict, time_hours: int):
        """Render search analytics visualizations"""
        
        st.subheader(f"🔍 Search Analytics - Last {time_hours} Hours")
        
        if analytics.get("total_searches", 0) == 0:
            st.info("No search data available for the selected time period")
            return
        
        # Search activity over time
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🕐 Search Activity by Hour**")
            hourly_data = analytics.get("hourly_distribution", {})
            
            if hourly_data:
                hours = list(hourly_data.keys())
                counts = list(hourly_data.values())
                
                fig = px.bar(
                    x=hours,
                    y=counts,
                    title="Searches per Hour",
                    labels={"x": "Hour", "y": "Number of Searches"}
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hourly data available")
        
        with col2:
            st.markdown("**📊 Average Results per Search**")
            avg_results = analytics.get("avg_results_per_search", 0)
            
            # Create a gauge chart for average results
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = avg_results,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Avg Results"},
                delta = {'reference': 5},
                gauge = {
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 3], 'color': "lightgray"},
                        {'range': [3, 7], 'color': "yellow"},
                        {'range': [7, 10], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 8
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top queries and authors
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔥 Top Search Queries**")
            top_queries = analytics.get("top_queries", [])
            
            if top_queries:
                query_df = pd.DataFrame(top_queries, columns=["Query", "Count"])
                query_df = query_df.head(10)  # Top 10
                
                fig = px.bar(
                    query_df,
                    x="Count",
                    y="Query",
                    orientation='h',
                    title="Most Popular Queries"
                )
                fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No query data available")
        
        with col2:
            st.markdown("**👥 Most Searched Authors**")
            top_authors = analytics.get("top_authors", [])
            
            if top_authors:
                author_df = pd.DataFrame(top_authors, columns=["Author", "Mentions"])
                author_df = author_df.head(10)  # Top 10
                
                fig = px.pie(
                    author_df,
                    values="Mentions",
                    names="Author",
                    title="Author Popularity"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No author data available")
        
        # Source distribution
        st.markdown("**📂 Content Source Distribution**")
        source_dist = analytics.get("source_distribution", {})
        
        if source_dist:
            source_df = pd.DataFrame(list(source_dist.items()), columns=["Source", "Count"])
            
            fig = px.treemap(
                source_df,
                path=["Source"],
                values="Count",
                title="Content Sources"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No source distribution data available")
    
    def _render_content_analytics(self, analytics: Dict):
        """Render content analytics visualizations"""
        
        st.subheader("📚 Content Insights")
        
        if analytics.get("total_content_items", 0) == 0:
            st.info("No content data available")
            return
        
        # Content overview metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Content Items",
                analytics.get("total_content_items", 0)
            )
        
        with col2:
            st.metric(
                "Average Text Length",
                f"{analytics.get('avg_text_length', 0)} chars"
            )
        
        with col3:
            st.metric(
                "Total Search Hits",
                analytics.get("total_search_hits", 0)
            )
        
        # Top content
        st.markdown("**🏆 Most Popular Content**")
        top_content = analytics.get("top_content", [])
        
        if top_content:
            content_df = pd.DataFrame(top_content)
            
            # Display as a table with metrics
            st.dataframe(
                content_df[["author", "source", "search_hits", "text_preview"]],
                column_config={
                    "author": "Author",
                    "source": "Source", 
                    "search_hits": st.column_config.NumberColumn("Search Hits", format="%d"),
                    "text_preview": "Content Length (chars)"
                },
                use_container_width=True
            )
        
        # Author statistics
        st.markdown("**👥 Author Statistics**")
        author_stats = analytics.get("author_stats", {})
        
        if author_stats:
            author_data = []
            for author, stats in author_stats.items():
                author_data.append({
                    "Author": author,
                    "Posts": stats.get("posts", 0),
                    "Total Hits": stats.get("total_hits", 0),
                    "Avg Hits/Post": stats.get("avg_hits_per_post", 0),
                    "Avg Text Length": stats.get("avg_text_length", 0)
                })
            
            author_df = pd.DataFrame(author_data)
            author_df = author_df.sort_values("Total Hits", ascending=False).head(15)
            
            # Create visualization
            fig = px.scatter(
                author_df,
                x="Posts",
                y="Avg Hits/Post", 
                size="Total Hits",
                hover_name="Author",
                title="Author Engagement Analysis",
                labels={
                    "Posts": "Number of Posts",
                    "Avg Hits/Post": "Average Hits per Post"
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed author table
        if author_stats:
            with st.expander("📋 Detailed Author Statistics"):
                st.dataframe(
                    author_df,
                    column_config={
                        "Total Hits": st.column_config.NumberColumn("Total Hits", format="%d"),
                        "Avg Hits/Post": st.column_config.NumberColumn("Avg Hits/Post", format="%.1f"),
                        "Avg Text Length": st.column_config.NumberColumn("Avg Text Length", format="%d")
                    },
                    use_container_width=True
                )
    
    def _render_system_analytics(self, analytics: Dict):
        """Render system performance analytics"""
        
        st.subheader("🖥️ System Performance")
        
        if not analytics or analytics.get("status") == "No system metrics available":
            st.info("No system performance data available")
            return
        
        # System overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            memory_mb = analytics.get("current_memory_mb", 0)
            st.metric(
                "Memory Usage",
                f"{memory_mb:.1f} MB",
                delta="🟢" if memory_mb < 1000 else "🟡" if memory_mb < 2000 else "🔴"
            )
        
        with col2:
            cpu_percent = analytics.get("current_cpu_percent", 0)
            st.metric(
                "CPU Usage", 
                f"{cpu_percent:.1f}%",
                delta="🟢" if cpu_percent < 50 else "🟡" if cpu_percent < 80 else "🔴"
            )
        
        with col3:
            cache_rate = analytics.get("current_cache_hit_rate", 0)
            st.metric(
                "Cache Hit Rate",
                f"{cache_rate:.1f}%",
                delta="🟢" if cache_rate > 80 else "🟡" if cache_rate > 50 else "🔴"
            )
        
        with col4:
            response_time = analytics.get("avg_response_time_ms", 0)
            st.metric(
                "Avg Response Time",
                f"{response_time:.0f}ms",
                delta="🟢" if response_time < 500 else "🟡" if response_time < 1500 else "🔴"
            )
        
        # Performance trends (simulated data for now)
        st.markdown("**📈 Performance Trends**")
        
        # Create sample trend data
        hours = list(range(24))
        memory_trend = [analytics.get("avg_memory_mb", 500) + np.random.normal(0, 50) for _ in hours]
        cpu_trend = [analytics.get("avg_cpu_percent", 30) + np.random.normal(0, 10) for _ in hours]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Memory Usage (MB)', 'CPU Usage (%)'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=hours, y=memory_trend, name="Memory", line=dict(color="blue")),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=hours, y=cpu_trend, name="CPU", line=dict(color="red")),
            row=2, col=1
        )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title_text="Hours Ago", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
        
        # System health indicators
        st.markdown("**🏥 System Health**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            health_score = self._calculate_health_score(analytics)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = health_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "System Health Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**System Status Indicators**")
            
            # Performance indicators
            indicators = [
                ("Memory", memory_mb < 1500, f"{memory_mb:.1f} MB"),
                ("CPU", cpu_percent < 70, f"{cpu_percent:.1f}%"),
                ("Cache", cache_rate > 70, f"{cache_rate:.1f}%"),
                ("Response Time", response_time < 1000, f"{response_time:.0f}ms")
            ]
            
            for name, is_good, value in indicators:
                status = "✅" if is_good else "⚠️"
                st.write(f"{status} **{name}**: {value}")
    
    def _render_clustering_analytics(self):
        """Render semantic clustering and topic modeling results"""
        
        st.subheader("🧩 Semantic Clustering & Topic Analysis")
        
        # Get current content for analysis
        if 'current_search_results' not in st.session_state:
            st.info("No content available for clustering analysis. Perform a search first to see clustering results.")
            return
        
        current_results = st.session_state.get('current_search_results', [])
        
        if len(current_results) < 3:
            st.info("Need at least 3 content items for meaningful clustering analysis.")
            return
        
        # Perform clustering analysis
        with st.spinner("Analyzing content clusters and topics..."):
            try:
                clustering_results = self.clustering_engine.analyze_content(current_results)
                
                if "clustering_success" in clustering_results and clustering_results["clustering_success"]:
                    self._display_clustering_results(clustering_results)
                else:
                    st.error(f"Clustering analysis failed: {clustering_results.get('clustering_error', 'Unknown error')}")
                
            except Exception as e:
                st.error(f"Error during clustering analysis: {str(e)}")
    
    def _display_clustering_results(self, results: Dict[str, Any]):
        """Display clustering analysis results"""
        
        # Clustering results
        if "clusters" in results:
            clusters = results["clusters"]
            st.markdown("**🎯 Content Clusters**")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"Found **{clusters.get('n_clusters', 0)} clusters** using **{clusters.get('method_used', 'unknown')}** method")
                st.write(f"Silhouette Score: **{clusters.get('silhouette_score', 0):.3f}** (higher is better)")
            
            with col2:
                if st.button("🔄 Re-analyze Clusters"):
                    st.rerun()
            
            # Display individual clusters
            cluster_data = clusters.get("clusters", {})
            
            if cluster_data:
                for cluster_id, cluster_info in cluster_data.items():
                    with st.expander(f"📊 {cluster_info['label']} ({cluster_info['post_count']} items)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Description:**")
                            st.write(cluster_info['description'])
                            
                            st.write("**Keywords:**")
                            keywords_str = ", ".join(cluster_info['keywords'][:5])
                            st.code(keywords_str)
                        
                        with col2:
                            st.metric("Coherence Score", f"{cluster_info['coherence_score']:.3f}")
                            st.write("**Sample Posts:**")
                            for post_id in cluster_info.get('sample_posts', [])[:3]:
                                st.write(f"• {post_id}")
            
        # Topic modeling results
        if "topics" in results and results.get("topic_modeling_success", False):
            st.markdown("---")
            st.markdown("**📚 Topic Analysis**")
            
            topics = results["topics"]
            topic_data = topics.get("topics", {})
            
            if topic_data:
                # Topic overview
                st.write(f"Discovered **{topics.get('n_topics', 0)} topics** across the content")
                
                # Create topic visualization
                topic_names = []
                topic_prevalence = []
                topic_coherence = []
                
                for topic_id, topic_info in topic_data.items():
                    topic_names.append(topic_info['label'])
                    topic_prevalence.append(topic_info['prevalence'])
                    topic_coherence.append(topic_info['coherence'])
                
                # Topic prevalence chart
                fig = px.bar(
                    x=topic_names,
                    y=topic_prevalence,
                    title="Topic Prevalence (%)",
                    labels={"x": "Topics", "y": "Prevalence (%)"}
                )
                fig.update_layout(height=300, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed topic information
                for topic_id, topic_info in topic_data.items():
                    with st.expander(f"📖 {topic_info['label']} ({topic_info['prevalence']:.1f}% of content)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Top Keywords:**")
                            for keyword, weight in topic_info['keywords']:
                                st.write(f"• {keyword} ({weight:.3f})")
                        
                        with col2:
                            st.metric("Coherence", f"{topic_info['coherence']:.3f}")
                            st.metric("Post Count", topic_info['post_count'])
        
        # Content insights
        if "insights" in results:
            st.markdown("---")
            st.markdown("**💡 Content Insights**")
            
            insights = results["insights"]
            quality = insights.get("quality_indicators", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Content Diversity", f"{quality.get('content_diversity', 0):.3f}")
                st.metric("Author Diversity", quality.get('author_diversity', 0))
            
            with col2:
                st.metric("Avg Text Length", f"{quality.get('avg_text_length', 0)} chars")
                st.metric("Source Diversity", quality.get('source_diversity', 0))
            
            with col3:
                st.metric("Total Posts Analyzed", quality.get('total_posts', 0))
            
            # Text length distribution
            text_stats = insights.get("text_length_stats", {})
            if text_stats:
                st.markdown("**📏 Text Length Distribution**")
                
                lengths = ["Min", "Median", "Average", "Max"]
                values = [
                    text_stats.get("min", 0),
                    text_stats.get("median", 0), 
                    text_stats.get("avg", 0),
                    text_stats.get("max", 0)
                ]
                
                fig = px.bar(
                    x=lengths,
                    y=values,
                    title="Text Length Statistics (characters)"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    def _calculate_health_score(self, analytics: Dict) -> float:
        """Calculate system health score based on performance metrics"""
        
        # Base score
        score = 100.0
        
        # Memory penalty
        memory_mb = analytics.get("current_memory_mb", 0)
        if memory_mb > 2000:
            score -= 30
        elif memory_mb > 1500:
            score -= 20
        elif memory_mb > 1000:
            score -= 10
        
        # CPU penalty
        cpu_percent = analytics.get("current_cpu_percent", 0)
        if cpu_percent > 80:
            score -= 25
        elif cpu_percent > 60:
            score -= 15
        elif cpu_percent > 40:
            score -= 5
        
        # Cache hit rate bonus/penalty
        cache_rate = analytics.get("current_cache_hit_rate", 0)
        if cache_rate > 90:
            score += 5
        elif cache_rate < 50:
            score -= 15
        elif cache_rate < 70:
            score -= 10
        
        # Response time penalty
        response_time = analytics.get("avg_response_time_ms", 0)
        if response_time > 2000:
            score -= 20
        elif response_time > 1000:
            score -= 10
        elif response_time > 500:
            score -= 5
        
        return max(0, min(100, score))


def render_analytics_dashboard():
    """Main function to render analytics dashboard"""
    dashboard = AnalyticsDashboard()
    dashboard.render_dashboard()